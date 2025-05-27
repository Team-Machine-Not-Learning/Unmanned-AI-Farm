<template>
  <n-space vertical size="large">
    <n-card title="实时视频与控制">
      <div class="video-container">
        <img v-if="isProcessing" :src="videoFeedUrl" alt="Video Feed" class="video-feed-img" />
        <div v-else class="video-placeholder">
          <n-icon size="60" :component="VideocamOffOutline" />
          <p>摄像头已关闭或未启动</p>
        </div>
      </div>
      <template #action>
        <n-space justify="center">
          <n-button type="primary" @click="startProcessing" :loading="isLoading" :disabled="isProcessing">
            开始灭虫
          </n-button>
          <n-button type="error" @click="stopProcessing" :loading="isLoading" :disabled="!isProcessing">
            停止灭虫
          </n-button>
          <n-button type="warning" @click="triggerLaser" :disabled="!isProcessing || !canTriggerLaser">
            触发激光 (模拟)
          </n-button>
        </n-space>
      </template>
    </n-card>
    <n-card title="检测信息">
      <n-descriptions label-placement="left" bordered :column="1">
        <n-descriptions-item label="当前目标">
          {{ currentTargetInfo.name || '无' }}
        </n-descriptions-item>
        <n-descriptions-item label="置信度">
          {{ currentTargetInfo.confidence ? (currentTargetInfo.confidence * 100).toFixed(2) + '%' : 'N/A' }}
        </n-descriptions-item>
        <n-descriptions-item label="目标坐标 (示例)">
          {{ currentTargetInfo.box ? currentTargetInfo.box.join(', ') : 'N/A' }}
        </n-descriptions-item>
      </n-descriptions>
    </n-card>
  </n-space>
</template>

<script setup>
import { ref, computed, onUnmounted } from 'vue';
import axios from 'axios';
import { NCard, NButton, NSpace, useMessage, NIcon, NDescriptions, NDescriptionsItem } from 'naive-ui';
import { VideocamOffOutline } from '@vicons/ionicons5';

const message = useMessage();

// 后端 API 的基础 URL (假设 FastAPI 运行在 RK3588 的 8000 端口)
// 在开发时，如果你的 Vite 开发服务器和 FastAPI 不在同一台机器或端口，你需要配置 Vite 的代理
// 或者确保 FastAPI 后端配置了正确的 CORS。
// 将 <RK3588_IP_ADDRESS> 替换为你的 RK3588 的实际 IP 地址。
const API_BASE_URL = 'http://localhost:8000'; // <--- IMPORTANT: CONFIGURE THIS

const videoFeedUrl = ref('');
const isProcessing = ref(false);
const isLoading = ref(false); // 用于按钮加载状态
const currentTargetInfo = ref({ name: null, confidence: null, box: null }); // 用于显示后端识别的最高置信度目标

// 模拟：后端可能会通过 WebSocket 或 SSE 推送此信息，或前端轮询
// 这里我们暂时不实现实时目标信息更新，仅在触发激光时从后端获取一次
// 为了简单，可以考虑修改后端 video_feed 的每一帧也附带目标信息（如果性能允许）
// 或者前端定期轮询一个 /laser/current_target API

// 计算属性，判断是否可以触发激光（例如，有目标时）
const canTriggerLaser = computed(() => !!currentTargetInfo.value.name);


async function startProcessing() {
  isLoading.value = true;
  try {
    const response = await axios.post(`${API_BASE_URL}/laser/start_processing`);
    if (response.data.status === 'success') {
      isProcessing.value = true;
      videoFeedUrl.value = `${API_BASE_URL}/laser/video_feed?t=${new Date().getTime()}`; // 加时间戳防止缓存
      message.success('激光灭虫程序已启动');
      // 后续可以轮询或通过 WebSocket 获取当前检测到的目标信息
    } else {
      message.warning(response.data.message || '启动失败');
    }
  } catch (error) {
    console.error('Error starting processing:', error);
    message.error('启动处理失败，请检查后端服务。');
  }
  isLoading.value = false;
}

async function stopProcessing() {
  isLoading.value = true;
  try {
    const response = await axios.post(`${API_BASE_URL}/laser/stop_processing`);
    isProcessing.value = false;
    videoFeedUrl.value = ''; // 清空视频流 URL
    currentTargetInfo.value = { name: null, confidence: null, box: null }; // 清空目标信息
    message.success(response.data.message || '激光灭虫程序已停止');
  } catch (error) {
    console.error('Error stopping processing:', error);
    message.error('停止处理失败，请检查后端服务。');
  }
  isLoading.value = false;
}

async function triggerLaser() {
  if (!isProcessing.value) {
    message.warning('请先启动灭虫程序');
    return;
  }
  // 这个API应该由后端提供，这里只是模拟前端调用
  try {
    const response = await axios.post(`${API_BASE_URL}/laser/trigger_action`);
    message.success(response.data.message || '激光指令已发送');
    if (response.data.target_info && typeof response.data.target_info === 'string') {
        // 简单解析后端返回的 target_info 字符串
        // 理想情况下后端应该返回结构化数据
        const nameMatch = response.data.target_info.match(/Target: ([\w-]+)/);
        const confMatch = response.data.target_info.match(/Conf: ([\d.]+)/);
        const boxMatch = response.data.target_info.match(/Box: \[([\d, ]+)\]/);

        if (nameMatch) currentTargetInfo.value.name = nameMatch[1];
        if (confMatch) currentTargetInfo.value.confidence = parseFloat(confMatch[1]) / 100; // Assuming backend sends as 0-100
        if (boxMatch) currentTargetInfo.value.box = boxMatch[1].split(',').map(s => parseInt(s.trim()));

    } else if (response.data.target_info && typeof response.data.target_info === 'object') {
        // 如果后端返回结构化数据
        currentTargetInfo.value = {
            name: response.data.target_info.name,
            confidence: response.data.target_info.confidence,
            box: response.data.target_info.box,
        };
    }


  } catch (error) {
    console.error('Error triggering laser:', error);
    message.error('触发激光失败');
  }
}

// 组件卸载时尝试停止处理，防止页面切换后视频流仍在后台请求
onUnmounted(() => {
  if (isProcessing.value) {
    stopProcessing();
  }
});

</script>

<style scoped>
.video-container {
  width: 100%;
  /* aspect-ratio: 16 / 9; */ /* 或者固定高度 */
  height: 480px; /* 根据你的大框需求调整 */
  background-color: #000; /* 视频未加载时的背景 */
  display: flex;
  justify-content: center;
  align-items: center;
  border: 1px solid #ccc;
  margin-bottom: 20px;
}

.video-feed-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain; /* 保持视频比例 */
}

.video-placeholder {
  color: #888;
  text-align: center;
}
</style>