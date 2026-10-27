package com.example.npu

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : Activity() {
    private lateinit var out: TextView
    private val npuModels = listOf("fc_256_2048_2048", "fc_1024_2048_2048", "fc_256_2048_8192")
    private val gpuModels = listOf(
        "attn_f32", "sm_int8", "qk_int8", "pv_int8",
        "fc_256_2048_2048", "fc_1024_2048_2048", "fc_256_2048_8192"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(24, 96, 24, 24)
        val b1 = Button(this)
        b1.text = "Run NPU test"
        val b2 = Button(this)
        b2.text = "Run GPU test"
        out = TextView(this)
        out.setTextIsSelectable(true)
        out.textSize = 13f
        val scroll = ScrollView(this)
        scroll.addView(out)
        root.addView(b1)
        root.addView(b2)
        root.addView(scroll)
        setContentView(root)
        b1.setOnClickListener { start(b1, b2) { runNpu() } }
        b2.setOnClickListener { start(b1, b2) { runGpu() } }
    }

    private fun start(b1: Button, b2: Button, job: () -> Unit) {
        b1.isEnabled = false
        b2.isEnabled = false
        out.text = ""
        Thread {
            try {
                job()
            } catch (t: Throwable) {
                log("ERROR " + Log.getStackTraceString(t))
            }
            runOnUiThread {
                b1.isEnabled = true
                b2.isEnabled = true
            }
        }.start()
    }

    private fun log(s: String) {
        Log.i("NPUTEST", s)
        runOnUiThread { out.append(s + "\n") }
    }

    private fun modelFile(name: String): File {
        val f = File(filesDir, "$name.tflite")
        if (!f.exists()) {
            assets.open("$name.tflite").use { i ->
                f.outputStream().use { o -> i.copyTo(o) }
            }
        }
        return f
    }

    private fun bench(interp: Interpreter): Double {
        val n = interp.inputTensorCount
        val inputs = Array<Any>(n) { i ->
            ByteBuffer.allocateDirect(interp.getInputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }
        val outputs = HashMap<Int, Any>()
        for (i in 0 until interp.outputTensorCount) {
            outputs[i] = ByteBuffer.allocateDirect(interp.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }
        fun once() {
            for (b in inputs) (b as ByteBuffer).rewind()
            for (b in outputs.values) (b as ByteBuffer).rewind()
            interp.runForMultipleInputsOutputs(inputs, outputs)
        }
        for (i in 0 until 3) once()
        val t = System.nanoTime()
        for (i in 0 until 20) once()
        return (System.nanoTime() - t) / 1e6 / 20
    }

    private fun runNpu() {
        val cache = File(filesDir, "nnapi")
        cache.mkdirs()
        log("cache files before: " + (cache.listFiles()?.size ?: 0))
        for (name in npuModels) {
            try {
                val f = modelFile(name)
                val t0 = System.nanoTime()
                val opts = NnApiDelegate.Options()
                opts.setAcceleratorName("mtk-mdla")
                opts.setCacheDir(cache.absolutePath)
                opts.setModelToken(name + "_v1")
                val nn = NnApiDelegate(opts)
                val io = Interpreter.Options()
                io.addDelegate(nn)
                val interp = Interpreter(f, io)
                val compileMs = (System.nanoTime() - t0) / 1e6
                val runMs = bench(interp)
                log(name + "  compile " + "%.1f".format(compileMs) + " ms  run " + "%.2f".format(runMs) + " ms")
                interp.close()
                nn.close()
            } catch (t: Throwable) {
                log(name + " ERROR " + t.toString())
            }
        }
        val files = cache.listFiles() ?: arrayOf()
        var total = 0L
        for (x in files) total += x.length()
        log("cache files after: " + files.size + "  bytes: " + total)
        log("done")
    }

    private fun runGpu() {
        for (name in gpuModels) {
            try {
                val f = modelFile(name)
                val ci = Interpreter(f, Interpreter.Options().setNumThreads(2))
                val cpuMs = bench(ci)
                ci.close()
                val t0 = System.nanoTime()
                val go = GpuDelegate.Options()
                go.setPrecisionLossAllowed(true)
                val gd = GpuDelegate(go)
                val io = Interpreter.Options()
                io.addDelegate(gd)
                val interp = Interpreter(f, io)
                val initMs = (System.nanoTime() - t0) / 1e6
                val gpuMs = bench(interp)
                log(name + "  cpu " + "%.2f".format(cpuMs) + " ms  gpu " + "%.2f".format(gpuMs) + " ms  gpu init " + "%.0f".format(initMs) + " ms")
                interp.close()
                gd.close()
            } catch (t: Throwable) {
                log(name + " ERROR " + t.toString())
            }
        }
        log("done")
    }
}
