/**
 * Canopy - Your Silent Interview Wingman
 * OpenRouter + Ollama Integration
 */

class CanopyDashboard {
    constructor() {
        this.isListening = false;
        this.recognition = null;
        this.transcriptHistory = [];
        this.lastQueryTime = 0;
        this.autoAnswer = true;
        this.isProcessing = false;
        this.settings = this.loadSettings();
        
        this.initElements();
        this.initEventListeners();
        this.initSpeechRecognition();
        this.updateModelInfo();
    }

    initElements() {
        this.toggleListenBtn = document.getElementById('toggleListenBtn');
        this.listenBtnText = document.getElementById('listenBtnText');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.transcriptArea = document.getElementById('transcriptArea');
        this.transcriptScroll = document.getElementById('transcriptScroll');
        this.questionInput = document.getElementById('questionInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.answerArea = document.getElementById('answerArea');
        this.answerContent = document.getElementById('answerContent');
        this.answerTime = document.getElementById('answerTime');
        this.emptyState = document.getElementById('emptyState');
        this.providerSelect = document.getElementById('providerSelect');
        this.modelInfo = document.getElementById('modelInfo');
        this.micLevelBar = document.querySelector('#micLevel div');
        
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettings = document.getElementById('closeSettings');
        this.cancelSettings = document.getElementById('cancelSettings');
        this.saveSettings = document.getElementById('saveSettings');
        
        this.captureScreenBtn = document.getElementById('captureScreenBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.autoAnswerCheck = document.getElementById('autoAnswerCheck');
        this.ollamaCustom = document.getElementById('ollamaCustom');
    }

    initEventListeners() {
        this.toggleListenBtn.addEventListener('click', () => this.toggleListening());
        this.sendBtn.addEventListener('click', () => this.handleTextQuestion());
        this.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.handleTextQuestion();
            }
        });
        
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        this.cancelSettings.addEventListener('click', () => this.closeSettingsModal());
        this.saveSettings.addEventListener('click', () => this.saveSettingsModal());
        
        this.captureScreenBtn.addEventListener('click', () => this.captureScreen());
        this.clearBtn.addEventListener('click', () => this.clearHistory());
        this.autoAnswerCheck.addEventListener('change', (e) => this.autoAnswer = e.target.checked);
        this.providerSelect.addEventListener('change', () => this.updateModelInfo());
        
        document.getElementById('ollamaModel').addEventListener('change', (e) => {
            this.ollamaCustom.classList.toggle('hidden', e.target.value !== 'custom');
        });
        
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettingsModal();
        });
    }

    updateModelInfo() {
        const provider = this.providerSelect.value;
        this.modelInfo.textContent = provider === 'openrouter' ? 'Uses your own API key' : 'Requires Ollama running locally';
    }

    initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.showToast('Web Speech API not available. Use Chrome or Edge.', 'warning');
            return;
        }
        
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = this.settings.language || 'en-US';
        
        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateStatus('listening');
        };
        
        this.recognition.onend = () => {
            if (this.isListening) {
                try { this.recognition.start(); } catch(e) {}
            } else {
                this.updateStatus('idle');
            }
        };
        
        this.recognition.onerror = (e) => {
            if (e.error === 'not-allowed') {
                this.updateStatus('error');
                this.showToast('Microphone access denied. Check browser permissions.', 'error');
            }
        };
        
        this.recognition.onresult = (e) => {
            for (let i = e.resultIndex; i < e.results.length; i++) {
                const transcript = e.results[i][0].transcript;
                if (e.results[i].isFinal) {
                    this.addTranscript('Interviewer', transcript);
                    if (this.autoAnswer && transcript.trim().length > 4) {
                        this.autoTriggerAnswer(transcript);
                    }
                }
            }
        };
    }

    toggleListening() {
        if (!this.recognition) {
            this.showToast('Speech recognition not supported in this browser', 'error');
            return;
        }
        
        if (this.isListening) {
            this.isListening = false;
            this.recognition.stop();
            this.updateStatus('idle');
        } else {
            this.isListening = true;
            try { this.recognition.start(); } catch(e) {}
            this.updateStatus('listening');
            if (this.emptyState) this.emptyState.style.display = 'none';
        }
    }

    updateStatus(state) {
        this.statusDot.className = 'status-indicator';
        const states = {
            listening: { class: 'status-listening', text: 'Listening', btnText: 'Stop', btnClass: 'btn-danger' },
            idle: { class: 'status-idle', text: 'Idle', btnText: 'Start Listening', btnClass: 'btn-primary' },
            error: { class: 'status-error', text: 'Error', btnText: 'Retry', btnClass: 'btn-primary' }
        };
        
        const config = states[state] || states.idle;
        this.statusDot.classList.add(config.class);
        this.statusText.textContent = config.text;
        this.listenBtnText.textContent = config.btnText;
        this.toggleListenBtn.classList.remove('btn-primary', 'btn-danger');
        this.toggleListenBtn.classList.add(config.btnClass);
    }

    addTranscript(speaker, text) {
        if (this.emptyState) this.emptyState.style.display = 'none';
        
        const isInterviewer = speaker === 'Interviewer';
        const div = document.createElement('div');
        div.className = 'flex gap-3 animate-fade-in';
        div.innerHTML = `
            <div class="w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0 mt-0.5 ${isInterviewer ? 'bg-amber-500/10 text-amber-400' : 'bg-blue-500/10 text-blue-400'}">
                ${isInterviewer ? 'I' : 'Y'}
            </div>
            <div class="flex-1 min-w-0">
                <div class="text-[10px] font-semibold ${isInterviewer ? 'text-amber-400/80' : 'text-blue-400/80'} mb-0.5 tracking-wide uppercase">${speaker}</div>
                <div class="text-[13px] text-gray-300 leading-relaxed">${this.escapeHtml(text)}</div>
            </div>
        `;
        this.transcriptArea.appendChild(div);
        this.scrollToBottom();
        this.transcriptHistory.push({ speaker, text, time: Date.now() });
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const colors = { info: 'bg-primary/90', success: 'bg-green-500/90', warning: 'bg-yellow-500/90', error: 'bg-red-500/90' };
        toast.className = `fixed bottom-20 left-1/2 -translate-x-1/2 ${colors[type]} text-white text-[11px] font-medium px-4 py-2 rounded-lg shadow-lg z-50 animate-fade-in`;
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    clearHistory() {
        this.transcriptArea.innerHTML = '';
        this.transcriptHistory = [];
        this.answerArea.classList.add('hidden');
        this.emptyState = document.createElement('div');
        this.emptyState.id = 'emptyState';
        this.emptyState.className = 'flex flex-col items-center justify-center py-20 text-gray-600';
        this.emptyState.innerHTML = `
            <div class="w-12 h-12 rounded-xl bg-surface/40 border border-white/[0.03] flex items-center justify-center mb-3">
                <svg class="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
            </div>
            <p class="text-sm font-medium text-gray-500">Ready to listen</p>
            <p class="text-[11px] text-gray-600 mt-0.5">Click Start Listening or type a question</p>
        `;
        this.transcriptArea.appendChild(this.emptyState);
    }

    scrollToBottom() {
        this.transcriptScroll.scrollTop = this.transcriptScroll.scrollHeight;
    }

    async handleTextQuestion() {
        const question = this.questionInput.value.trim();
        if (!question || this.isProcessing) return;
        
        this.questionInput.value = '';
        if (this.emptyState) this.emptyState.style.display = 'none';
        this.addTranscript('Candidate', question);
        await this.getAnswer(question);
    }

    async autoTriggerAnswer(text) {
        const now = Date.now();
        if (now - this.lastQueryTime < 2000) return;
        this.lastQueryTime = now;
        await this.getAnswer(text);
    }

    async getAnswer(question) {
        if (this.isProcessing) return;
        this.isProcessing = true;
        
        this.answerArea.classList.remove('hidden');
        this.answerContent.innerHTML = `
            <div class="flex items-center space-x-2 text-gray-500">
                <svg class="w-3.5 h-3.5 animate-spin text-primary" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span class="text-[12px]">Generating answer...</span>
            </div>`;
        this.answerTime.textContent = '';
        this.scrollToBottom();
        
        const startTime = Date.now();
        
        try {
            const provider = this.providerSelect.value;
            let response;
            
            if (provider === 'openrouter') {
                response = await this.callOpenRouter(question);
            } else {
                response = await this.callOllama(question);
            }
            
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            this.answerTime.textContent = `${elapsed}s`;
            this.renderAnswer(response);
        } catch (e) {
            this.answerContent.innerHTML = `<div class="text-red-400 text-[12px]">Error: ${this.escapeHtml(e.message)}</div>`;
            this.showToast(e.message, 'error');
        } finally {
            this.isProcessing = false;
        }
    }

    async callOpenRouter(question) {
        const apiKey = this.settings.openrouterKey;
        if (!apiKey) {
            throw new Error('OpenRouter API key not configured. Add it in Settings.');
        }
        
        const model = this.settings.openrouterModel || 'google/gemma-2-9b-it:free';
        
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
                'HTTP-Referer': window.location.href,
                'X-Title': 'Canopy Interview Copilot'
            },
            body: JSON.stringify({
                model: model,
                messages: [
                    {
                        role: 'system',
                        content: 'You are an expert interview coach. Provide concise, optimal solutions to coding interview questions. Include time/space complexity analysis and key talking points. Keep responses under 300 words unless the problem demands more detail.'
                    },
                    { role: 'user', content: question }
                ],
                max_tokens: 800,
                temperature: 0.3
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error?.message || `OpenRouter error: ${response.status}`);
        }
        
        const data = await response.json();
        return data.choices[0].message.content;
    }

    async callOllama(question) {
        const url = this.settings.ollamaUrl || 'http://localhost:11434';
        let model = this.settings.ollamaModel || 'gemma2';
        if (model === 'custom') model = this.settings.ollamaCustom || 'gemma2';
        
        const response = await fetch(`${url}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model,
                messages: [
                    {
                        role: 'system',
                        content: 'You are an expert interview coach. Provide concise, optimal solutions to coding interview questions. Include time/space complexity analysis and key talking points. Keep responses under 300 words unless the problem demands more detail.'
                    },
                    { role: 'user', content: question }
                ],
                stream: false,
                options: { temperature: 0.3 }
            })
        });
        
        if (!response.ok) {
            throw new Error(`Ollama not responding. Is it running at ${url}? Run: ollama run ${model}`);
        }
        
        const data = await response.json();
        return data.message?.content || data.response || 'No response from model';
    }

    renderAnswer(text) {
        let html = this.escapeHtml(text)
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre class="bg-dark/80 p-3 rounded-lg mt-2 mb-2 overflow-x-auto border border-white/[0.04]"><code class="text-[12px] font-mono text-gray-300 leading-relaxed">$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code class="bg-dark/60 px-1 py-0.5 rounded text-[11px] text-secondary/90 font-mono">$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
            .replace(/^### (.*$)/gim, '<h4 class="text-white font-semibold mt-3 mb-1 text-[13px]">$1</h4>')
            .replace(/^## (.*$)/gim, '<h3 class="text-white font-bold mt-4 mb-2 text-[14px]">$1</h3>')
            .replace(/^# (.*$)/gim, '<h2 class="text-white font-bold mt-4 mb-2 text-[15px]">$1</h2>')
            .replace(/\n/g, '<br>');
        this.answerContent.innerHTML = html;
        this.scrollToBottom();
    }

    async captureScreen() {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
            const track = stream.getVideoTracks()[0];
            const imageCapture = new ImageCapture(track);
            const bitmap = await imageCapture.grabFrame();
            
            const canvas = document.createElement('canvas');
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0);
            
            track.stop();
            this.showToast('Screen captured', 'success');
        } catch (e) {
            this.showToast('Screen capture cancelled', 'info');
        }
    }

    openSettings() {
        document.getElementById('openrouterKey').value = this.settings.openrouterKey || '';
        document.getElementById('openrouterModel').value = this.settings.openrouterModel || 'google/gemma-2-9b-it:free';
        document.getElementById('ollamaUrl').value = this.settings.ollamaUrl || 'http://localhost:11434';
        document.getElementById('ollamaModel').value = this.settings.ollamaModel || 'gemma2';
        document.getElementById('languageSelect').value = this.settings.language || 'en-US';
        this.ollamaCustom.classList.toggle('hidden', this.settings.ollamaModel !== 'custom');
        this.settingsModal.classList.remove('hidden');
    }

    closeSettingsModal() {
        this.settingsModal.classList.add('hidden');
    }

    saveSettingsModal() {
        this.settings = {
            openrouterKey: document.getElementById('openrouterKey').value.trim(),
            openrouterModel: document.getElementById('openrouterModel').value,
            ollamaUrl: document.getElementById('ollamaUrl').value.trim(),
            ollamaModel: document.getElementById('ollamaModel').value,
            ollamaCustom: document.getElementById('ollamaCustom').value.trim(),
            language: document.getElementById('languageSelect').value,
        };
        localStorage.setItem('canopySettings', JSON.stringify(this.settings));
        
        if (this.recognition) {
            this.recognition.lang = this.settings.language || 'en-US';
        }
        
        this.closeSettingsModal();
        this.showToast('Settings saved', 'success');
    }

    loadSettings() {
        try {
            return JSON.parse(localStorage.getItem('canopySettings')) || {};
        } catch {
            return {};
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CanopyDashboard();
});
