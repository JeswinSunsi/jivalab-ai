import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    allowedHosts: ['localhost', '4525-2401-4900-6143-3924-a562-b7bb-61d6-ffb2.ngrok-free.app', "*", "0.0.0.0", "*.ngrok-free.app"],
    host: true, // This enables listening on all local IPs
  }
})