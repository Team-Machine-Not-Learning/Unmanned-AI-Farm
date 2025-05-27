<template>
  <n-config-provider :theme-vars="themeVars">
    <n-message-provider>
      <n-layout style="height: 100vh">
        <!-- 固定侧边栏 -->
        <n-layout-sider
          bordered
          collapse-mode="width"
          :collapsed-width="64"
          :width="240"
          show-trigger
          inverted
          :native-scrollbar="false"
          style="position: fixed; height: 100vh;"
        >
          <n-menu
            :inverted="true"
            :collapsed-width="64"
            :collapsed-icon-size="22"
            :options="menuOptions"
            v-model:value="activeMenuKey"
            @update:value="handleMenuSelect"
          />
        </n-layout-sider>

        <!-- 主内容区域 -->
        <n-layout :style="contentStyle">
          <!-- 固定顶部导航 -->
          <n-layout-header
            bordered
            style="height: 64px; padding: 0 24px; position: sticky; top: 0; z-index: 1000;"
          >
            <div style="display: flex; align-items: center; height: 100%; font-size: 1.5em;">
              {{ currentTitle }}
            </div>
          </n-layout-header>

          <!-- 可滚动内容区域 -->
          <n-layout-content style="padding: 24px;">
            <div style="max-width: 1200px; margin: 0 auto;">
              <LaserExtermination v-if="activeMenuKey === 'laser-extermination'" />
              <StatusPanel v-if="activeMenuKey === 'status-panel'" />
              <Encyclopedia v-if="activeMenuKey === 'encyclopedia'" />
              <div v-if="activeMenuKey === 'settings'">设置页面 (占位)</div>
              <div v-if="activeMenuKey === 'logs'">日志页面 (占位)</div>
              <div v-if="activeMenuKey === 'about'">关于页面 (占位)</div>
            </div>
          </n-layout-content>
        </n-layout>
      </n-layout>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup>
import { ref, computed, h } from 'vue';
import { 
  NLayout, 
  NLayoutSider, 
  NLayoutHeader, 
  NLayoutContent, 
  NMenu, 
  NIcon, 
  NConfigProvider, 
  NMessageProvider 
} from 'naive-ui';
import { 
  BarChartOutline, 
  BugOutline, 
  BulbOutline, 
  SettingsOutline, 
  DocumentTextOutline, 
  InformationCircleOutline 
} from '@vicons/ionicons5';

// 导入子组件
import LaserExtermination from './components/LaserExtermination.vue';
import StatusPanel from './components/StatusPanel.vue';
import Encyclopedia from './components/Encyclopedia.vue';

const activeMenuKey = ref('laser-extermination');

// 响应式内容区域样式
const contentStyle = computed(() => ({
  marginLeft: '240px',
  transition: 'margin-left .3s cubic-bezier(.4,0,.2,1)',
  minHeight: '100vh'
}));

// 主题变量（可选）
const themeVars = {
  common: {
    // primaryColor: '#2080F0',
    // primaryColorHover: '#4098FC',
  },
};

// 菜单图标渲染
function renderIcon(icon) {
  return () => h(NIcon, null, { default: () => h(icon) });
}

// 菜单配置
const menuOptions = [
  { label: '激光灭虫', key: 'laser-extermination', icon: renderIcon(BugOutline) },
  { label: '状态面板', key: 'status-panel', icon: renderIcon(BarChartOutline) },
  { label: '昆虫百科', key: 'encyclopedia', icon: renderIcon(BulbOutline) },
  { label: '系统设置', key: 'settings', icon: renderIcon(SettingsOutline) },
  { label: '运行日志', key: 'logs', icon: renderIcon(DocumentTextOutline) },
  { label: '关于系统', key: 'about', icon: renderIcon(InformationCircleOutline) },
];

// 当前标题计算
const currentTitle = computed(() => 
  menuOptions.find(option => option.key === activeMenuKey.value)?.label || '昆虫灭杀系统'
);

// 菜单选择处理
function handleMenuSelect(key) {
  activeMenuKey.value = key;
}
</script>

<style>
/* 全局样式 */
body, html, #app {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'Lato', sans-serif;
  overflow: hidden; /* 防止整体页面滚动 */
}

/* 侧边栏固定样式 */
.n-layout-sider {
  position: fixed !important;
  left: 0;
  top: 0;
  bottom: 0;
  height: 100vh;
  z-index: 1001;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
  transition: width .3s cubic-bezier(.4,0,.2,1);
}

/* 主内容区域适配 */
.n-layout-scroll-container {
  overflow-y: auto;
  height: 100vh;
}

/* 折叠状态适配 */
.n-layout-sider--collapsed + .n-layout {
  margin-left: 64px !important;
}

/* 顶部导航固定 */
.n-layout-header {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: var(--n-color);
  backdrop-filter: saturate(180%) blur(5px);
}
</style>