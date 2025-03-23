<template>
  <div class="jivalab-container">
    <header class="header">
      <span style="display: flex; align-items: center;">
        <img src="../assets/arrowblue.png" alt="Go Back" class="arrow-back" @click="$router.push('/home')">
        <div class="logo">Jiva<span style="font-weight: 400;">lab</span></div>
      </span>
      <div class="header-icons">
        <div class="notification-icon">
          <img src="../assets/chat.png" alt="Notifications" class="icon-placeholder" @click="$router.push('/chat')">
        </div>
        <div class="profile-icon">
          <img src="../assets/usericon.png" alt="Notifications" class="icon-placeholder" @click="$router.push('/profile')">
        </div>
      </div>
    </header>

    <div class="content-card">
      <div class="red-top"></div>
      <div class="mic-icon-container" @click="triggerFileInput">
        <div class="mic-icon">
          <img src="../assets/upload.png" alt="Upload" class="mic-icon-placeholder">
        </div>
      </div>

      <input 
        type="file" 
        ref="fileInput" 
        @change="handleFileChange" 
        accept="image/jpeg,image/png" 
        style="display: none;" 
      />

      <div class="instructions-list">
        <div class="instruction-item" v-motion
          :initial="{ x: -50, opacity: 0 }"
          :enter="{ x: 0, opacity: 1, transition: { delay: 100 } }">
          <div class="instruction-number">1</div>
          <div class="instruction-text">
            <p style="color: #0896B6;">Supported files<span class="highlight"> include JPG, PNG</span></p>
          </div>
        </div>
      </div>

      <div v-if="selectedFileName" class="selected-file">
        <p>Selected: {{ selectedFileName }}</p>
      </div>

      <button 
        class="record-button" 
        @click="uploadImage" 
        :disabled="!hasImage || isLoading"
        :class="{ 'button-disabled': !hasImage || isLoading }" v-motion
          :initial="{ x: -50, opacity: 0 }"
          :enter="{ x: 0, opacity: 1, transition: { delay: 300 } }"
      >
        {{ isLoading ? 'Uploading...' : 'My files are ready' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

import { useRoute } from "vue-router"
const route = useRoute()
const disease = route.params.disease

const fileInput = ref(null);
const hasImage = ref(false);
const isLoading = ref(false);
const apiResponse = ref(null);
const selectedFileName = ref('');

const triggerFileInput = () => {
  fileInput.value.click();
};

const handleFileChange = (event) => {
  const file = event.target.files[0];
  if (file) {
    hasImage.value = true;
    selectedFileName.value = file.name;
  } else {
    hasImage.value = false;
    selectedFileName.value = '';
  }
};

const uploadImage = async () => {
  if (!hasImage.value || isLoading.value) return;
  
  const file = fileInput.value.files[0];
  if (!file) return;
  
  try {
    isLoading.value = true;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`https://305d-14-139-184-222.ngrok-free.app/predict/${disease}`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    
    // Parse and store the JSON response
    const data = await response.json();
    apiResponse.value = data;
    console.log(apiResponse.value);
    
  } catch (error) {
    console.error('Error uploading image:', error);
    apiResponse.value = { error: error.message };
    alert('Error uploading image: ' + error.message);
  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
.jivalab-container {
  font-family: Poppins;
  max-width: 414px;
  margin: 0 auto;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background-color: #f5f8f9;
  margin-bottom: 1.75rem;
}

.logo {
  color: #0896B6;
  font-weight: bold;
  font-size: 22px;
}

.header-icons {
  display: flex;
  gap: 16px;
}

.arrow-back {
  transform: scaleX(-1);
  -webkit-transform: scaleX(-1);
  height: 2rem;
  width: auto;
  margin-right: 0.6rem;
}

.notification-button, .profile-button {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 20px;
}

.content-card {
  background-color: #fff;
  border-radius: 10px;
  margin: 16px;
  padding-bottom: 0.9rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  border: #0896B6 0.5px solid;
}

.red-top {
  width: 100%;
  height: 2.5rem;
  background-color: #0896B6;
  border-top-left-radius: 0.6rem;
  border-top-right-radius: 0.6rem;
  margin-bottom: 2.5rem;
}

.icon-placeholder {
  height: 1.5rem;
  width: auto;
}

.mic-icon-placeholder {
  height: 2.8rem;
  width: auto;
}

.mic-icon-container {
  background-color: rgba(99, 180, 121, 0.1);
  width: 7rem;
  height: 7rem;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 16px;
  cursor: pointer;
}

.title {
  color: #0896B6;
  font-size: 1.1rem;
  margin-bottom: 0.1rem;
  margin-top: 1rem;
  font-weight: 500;
  text-align: center;
}

.instructions-list {
  width: 93%;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 24px;
}

.instruction-item {
  display: flex;
  background-color: #F0F8FF;
  align-items: center;
  border-radius: 0.5rem;
  padding: 12px 1rem;
}

.instruction-number {
  background-color: #0896B6;
  color: #FFF;
  font-weight: 400;
  width: 24px;
  height: 24px;
  font-size: 0.8rem;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-right: 12px;
}

.instruction-text {
  flex: 1;
  font-size: 0.85rem;
}

.instruction-text p {
  margin: 0;
  line-height: 1.2rem;
}

.highlight {
  color: #009CB4;
  font-weight: 600;
}

.selected-file {
  margin: 0 0 16px 0;
  width: 100%;
  text-align: center;
  font-size: 0.9rem;
  color: #0896B6;
}

.record-button {
  background-color: #009CB4;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 16px 24px;
  font-size: 1.1rem;
  font-weight: bold;
  width: 93%;
  cursor: pointer;
  transition: background-color 0.2s;
}

.record-button:hover:not(:disabled) {
  background-color: #008CA4;
}

.button-disabled {
  background-color: #a0d3dd;
  cursor: not-allowed;
}
</style>