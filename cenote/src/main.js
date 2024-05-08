import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import "@cyrilf/vue-dat-gui/dist/style.css";
import VueDatGui from "@cyrilf/vue-dat-gui";
createApp(App).use(VueDatGui).mount('#app')
