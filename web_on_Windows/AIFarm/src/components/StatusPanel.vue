<template>
  <n-card title="状态面板">
    <p>这里将显示系统状态信息...</p>
    <n-descriptions label-placement="left" bordered :column="1">
      <n-descriptions-item label="RK3588 温度">
        {{ status.temperature || '加载中...' }}
      </n-descriptions-item>
      <n-descriptions-item label="系统负载">
         {{ status.system_load || '加载中...' }}
      </n-descriptions-item>
    </n-descriptions>
  </n-card>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { NCard, NDescriptions, NDescriptionsItem } from 'naive-ui';

const API_BASE_URL = 'http://<RK3588_IP_ADDRESS>:8000'; // <--- CONFIGURE THIS
const status = ref({});

async function fetchStatus() {
  try {
    const response = await axios.get(`${API_BASE_URL}/status_panel`);
    status.value = response.data;
  } catch (error) {
    console.error("Error fetching status:", error);
    status.value = { temperature: "获取失败", system_load: "获取失败" };
  }
}

onMounted(() => {
  fetchStatus();
  // Можно установить интервал для обновления статуса
  // setInterval(fetchStatus, 5000); // например, каждые 5 секунд
});
</script>