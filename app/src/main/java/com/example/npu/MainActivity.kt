package com.example.npu

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : Activity() {
    private lateinit var out: TextView
    private val models = listOf("fc_256_2048_2048", "fc_1024_2048_2048", "fc_256_2048_8192")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(24, 96, 24, 24)
        val btn = Button(this)
        btn.text = "Run test"
        out = TextView(this)
        out.setTextIsSelectable(true)
        out.textSize = 13f
        val scroll = ScrollView(this)
        scroll.addView(out)
        root.addView(btn)
        root.addView(scroll)
        setContentView(root)
        btn.setOnClickListener {
            btn.isEnabled = false
            out.text = ""
            Thread {
                try {
                    runAll()
                } catch (t: Throwable) {
                    log("ERROR " + Log.getStackTraceString(t))
                }
                runOnUiThread { btn.isEnabled = true }
            }.start()
        }
    }

    private fun log(s: String) {
        Log.i("NPUTEST", s)
        runOnUiThread { out.append(s + "\n") }
    }

    private fun runAll() {
        val cache = File(cacheDir, "nnapi")
        cache.mkdirs()
        log("cache files before: " + (cache.listFiles()?.size ?: 0))
        for (name in models) {
            val f = File(filesDir, "$name.tflite")
            if (!f.exists()) {
                assets.open("$name.tflite").use { i ->
                    f.outputStream().use { o -> i.copyTo(o) }
                }
            }
            val p = name.split("_")
            val tokens = p[1].toInt()
            val din = p[2].toInt()
            val dout = p[3].toInt()

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

            val input = ByteBuffer.allocateDirect(tokens * din).order(ByteOrder.nativeOrder())
            val output = ByteBuffer.allocateDirect(tokens * dout).order(ByteOrder.nativeOrder())
            for (i in 0 until 3) {
                input.rewind()
                output.rewind()
                interp.run(input, output)
            }
            val t1 = System.nanoTime()
            val n = 20
            for (i in 0 until n) {
                input.rewind()
                output.rewind()
                interp.run(input, output)
            }
            val runMs = (System.nanoTime() - t1) / 1e6 / n
            log(name + "  compile " + "%.1f".format(compileMs) + " ms  run " + "%.2f".format(runMs) + " ms")
            interp.close()
            nn.close()
        }
        val files = cache.listFiles() ?: arrayOf()
        var total = 0L
        for (x in files) total += x.length()
        log("cache files after: " + files.size + "  bytes: " + total)
        log("done")
    }
}
