import { createApp } from 'vue'
import App from './App.vue'
import naive from 'naive-ui'

// 通用字体
import 'vfonts/Lato.css'
// 等宽字体
import 'vfonts/FiraCode.css'

const app = createApp(App)
app.use(naive) // 全局注册 Naive UI 组件
app.mount('#app')