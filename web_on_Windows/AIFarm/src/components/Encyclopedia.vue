<template>
  <n-card title="昆虫百科">
    <n-input-group style="margin-bottom: 20px;">
      <n-input v-model:value="searchTerm" placeholder="输入昆虫名称搜索" />
      <n-button type="primary" @click="searchInsect">搜索</n-button>
    </n-input-group>
    <div v-if="insectInfo">
      <n-h3>{{ searchTermDisplay }}</n-h3>
      <n-descriptions label-placement="top" bordered :column="1">
        <n-descriptions-item label="描述">
          {{ insectInfo.description }}
        </n-descriptions-item>
        <n-descriptions-item label="防治方法">
          {{ insectInfo.control_methods }}
        </n-descriptions-item>
      </n-descriptions>
    </div>
    <n-empty v-if="searchPerformed && !insectInfo" description="未找到该昆虫信息" />
  </n-card>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import { NCard, NInputGroup, NInput, NButton, NDescriptions, NDescriptionsItem, NH3, NEmpty, useMessage } from 'naive-ui';

const API_BASE_URL = 'http://<RK3588_IP_ADDRESS>:8000'; // <--- CONFIGURE THIS
const searchTerm = ref('');
const searchTermDisplay = ref('');
const insectInfo = ref(null);
const searchPerformed = ref(false);
const message = useMessage();

async function searchInsect() {
  if (!searchTerm.value.trim()) {
    message.warning('请输入昆虫名称');
    return;
  }
  searchPerformed.value = true;
  insectInfo.value = null; // Reset previous info
  searchTermDisplay.value = searchTerm.value;
  try {
    const response = await axios.get(`${API_BASE_URL}/encyclopedia/${encodeURIComponent(searchTerm.value.trim())}`);
    insectInfo.value = response.data;
  } catch (error) {
    console.error("Error fetching encyclopedia entry:", error);
    if (error.response && error.response.status === 404) {
      message.error(`未找到 "${searchTermDisplay.value}" 的百科信息`);
    } else {
      message.error('查询百科失败');
    }
  }
}
</script>