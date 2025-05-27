import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // 将所有 /api 的请求代理到你的 FastAPI 后端
      // 例如：前端请求 /api/laser/start_processing 会被代理到 http://<RK3588_IP_ADDRESS>:8000/laser/start_processing
      '/api': {
        target: 'http://<RK3588_IP_ADDRESS>:8000', // <--- 你的 FastAPI 后端地址
        changeOrigin: true, // 需要虚拟主机站点
        rewrite: (path) => path.replace(/^\/api/, ''), // 去掉路径中的 /api 前缀
      },
    }
  }
})