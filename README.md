# 👥 Speaker ID / Diarization

**Container:** `speaker-id-diarization`  
**Ecossistema:** Mordomo  
**Posição no Fluxo:** Quinto - identificação de falantes

## 🚀 Quick Start

### Instalação

```bash
# Instalar dependências de teste
cd test_data
pip install -r requirements.txt
```

### Criar Embeddings de Teste

```bash
# Usuário 1 (você)
python test_data/create_embedding.py user_1

# Usuário 2 (outra pessoa)
python test_data/create_embedding.py user_2
```

### Testar Diarization

```bash
# Gravar áudio com múltiplos falantes e testar separação
python test_data/test_diarization.py --duration 10
```

Consulte `test_data/README.md` para detalhes completos dos testes.

---

## 📋 Propósito

Identificar QUEM está falando em cada momento (com embeddings cadastrados), separar conversas por falante e detectar sobreposição de vozes.

**Processamento Paralelo:** Inicia análise junto com Whisper ASR após `wake_word.detected`, mas só envia metadados downstream após `speaker.verified` (GATE). Reduz latência total em ~100-200ms.

**🔬 TECNICAMENTE:** Combina Diarization (separar vozes) + Speaker Recognition (identificar com embeddings):
- **Diarization puro** só agrupa: "Speaker 0, Speaker 1" (anônimos)
- **Speaker Recognition** identifica: compara embedding atual vs database cadastrado → "user_1", "user_2" ou "unknown"
- **Híbrido** = separa vozes + identifica cada uma com user_id específico

**🔒 CRÍTICO PARA SEGURANÇA:** Re-autentica continuamente durante conversação ativa, prevenindo escalação de privilégios quando falante muda (ex: admin inicia sessão, convidado tenta comando privilegiado).

---

## 🎯 Responsabilidades

- ✅ **Iniciar análise** após `wake_word.detected` (paralelo com Verification)
- ✅ **Receber áudio + transcrição** do Whisper ASR (gRPC)
- ✅ **Processar em buffer** até receber `speaker.verified` (GATE)
- ✅ **Identificar falante** (user_1, user_2, guest_123, etc)
- ✅ **Re-autenticar continuamente** (prevenir escalação de privilégios)
- ✅ **Detectar troca de falante** durante conversação ativa
- ✅ **Detectar sobreposição** de vozes
- ✅ **Publicar metadados** após gate aberto
- ✅ **Trigger Source Separation** se necessário
- ✅ **Descartar buffer** se `speaker.rejected`

---

## 🔄 Estados do Speaker ID

```
IDLE (aguardando)
  ↓ wake_word.detected
BUFFERING (processa mas não publica)
  ├─ speaker.verified → ANALYZING
  └─ speaker.rejected → IDLE (descarta buffer)
ANALYZING (publica resultados)
  ↓ conversation.ended (reseta gate)
IDLE (limpa contexto, pronto para próxima)
```

**Otimização:** Processa em paralelo com Verification (~200ms), resultados prontos quando gate abre.
**Reset:** `conversation.ended` limpa todo contexto e volta ao IDLE, resetando o gate para próxima detecção.

---

## 🔧 Tecnologias

**Linguagem:** Python (obrigatório - ecossistema ML)

**Principal:** pyannote.audio (Híbrido: Diarization + Recognition)
- Diarization state-of-the-art
- Speaker embeddings (256D)
- Overlap detection
- **Recognition:** Compara com embeddings cadastrados
- **Backend:** PyTorch (C++ libtorch nativo)

**Modelo:** `pyannote/speaker-diarization-3.1` + embeddings

**Database Compartilhado:**
- Usa MESMOS embeddings do Speaker Verification
- Embeddings cadastrados: `/data/embeddings/user_1.npy`, `/data/embeddings/user_2.npy`
- Threshold: 0.70 (mais permissivo que Verification)

**Alternativas:**
- Resemblyzer (mesma tech do Verification)
- SpeechBrain Speaker Recognition (PyTorch)
- Custom model (TensorFlow Lite)

**Performance:** Diarization model em PyTorch C++, embedding comparison em NumPy C (OpenBLAS). Python overhead ~10ms.

---

## 💾 Armazenamento de Embeddings (Compartilhado)

### Volume Compartilhado com Speaker Verification

```yaml
# Estrutura de diretórios
./data/embeddings/  (host - criado pelo Verification)
  └─ /data/embeddings/  (container - bind mount RO)

MESMOS arquivos do Verification:
  ├─ user_1.npy
  ├─ user_2.npy
  └─ guest_*.npy
```

**Modo de Acesso:** Read-Only (RO)
- Speaker Verification: Read-Write (cria/atualiza embeddings)
- Speaker ID/Diarization: Read-Only (apenas lê)

**Sincronização:**
- Embeddings criados pelo Verification são automaticamente visíveis
- Hot reload automático (watchdog detecta novos arquivos)
- Latência de leitura: ~0.5ms (cache do kernel)

**Docker Compose:**
```yaml
services:
  speaker-id-diarization:
    volumes:
      - ./data/embeddings:/data/embeddings:ro  # Read-Only
    environment:
      - EMBEDDINGS_PATH=/data/embeddings
```

**Vantagens:**
- ✅ Zero duplicação de dados
- ✅ Consistência automática (mesma fonte)
- ✅ Latência desprezível (< 1% do total)
- ✅ Cadastro centralizado (enrollment via Verification)

---

## 📊 Especificações

```yaml
Input:
  Audio + Transcript
  Sample Rate: 16000 Hz
  Duration: variável

Diarization:
  Min Speaker Duration: 1.0s
  Overlap Detection: true
  Max Speakers: 3
  
Recognition (identificação com embeddings):
  Database: /data/embeddings/*.npy (compartilhado com Verification)
  Enrolled Users: 2+ (user_1, user_2, guests)
  Threshold: 0.70 (cosine similarity)
  Unknown Detection: true
  Embedding Dimension: 256D
  
Performance:
  CPU: 30-50% spike
  RAM: ~ 800 MB
  Latency: < 300 ms
  Accuracy: > 90%
```

---

## 🔌 Interfaces

### Input (gRPC)
```protobuf
message DiarizeRequest {
  bytes audio = 1;
  string transcript = 2;
  int64 timestamp = 3;
}
```

### Output (NATS)
```python
# Voz reconhecida (cadastrada)
subject: "speech.diarized.{speaker_id}"
payload: {
  "text": "qual a temperatura",
  "speaker_id": "user_1",  # CRÍTICO: usado para re-autenticação
  "recognized": true,       # Voz encontrada no database
  "confidence": 0.88,       # Similaridade com embedding cadastrado
  "start_time": 0.5,
  "end_time": 2.3,
  "overlap_detected": false,
  "timestamp": 1732723200.123,
  "conversation_id": "abc123"
}

# Voz desconhecida (não cadastrada)
subject: "speech.diarized.unknown"
payload: {
  "text": "desliga o alarme",
  "speaker_id": "unknown_abc123",  # Hash do embedding
  "recognized": false,              # ⚠️ Voz NÃO encontrada
  "confidence": 0.42,               # Melhor match (< 0.70 threshold)
  "start_time": 5.0,
  "end_time": 7.5,
  "overlap_detected": false,
  "timestamp": 1732723205.123,
  "conversation_id": "abc123"
}

# Conversation Manager usa para:
#  1. Validar permissões do FALANTE ATUAL (não do dono da sessão)
#  2. IGNORAR comandos de recognized=false (vozes desconhecidas)
#  3. Detectar troca de falante (anti-escalação)
#  4. Manter contexto individualizado
#  5. Log de auditoria
```

---

## ⚙️ Configuração

```yaml
diarization:
  model: "pyannote/speaker-diarization-3.1"
  min_speaker_duration: 1.0
  max_speakers: 3
  
recognition:
  # Database compartilhado com Speaker Verification
  embeddings_path: "/data/embeddings"
  threshold: 0.70  # Cosine similarity
  unknown_detection: true
  
  # Usuários cadastrados
  enrolled_users:
    - user_id: "user_1"
      embedding_file: "user_1.npy"
    - user_id: "user_2"
      embedding_file: "user_2.npy"
  
overlap:
  detection_enabled: true
  threshold: 0.5  # Sobreposição temporal
  
source_separation:
  trigger_on_overlap: true
  min_overlap_duration: 0.5  # segundos

nats:
  url: "nats://nats:4222"
  publish_recognized: "speech.diarized.{speaker_id}"
  publish_unknown: "speech.diarized.unknown"
```

---

## 🔬 Implementação Técnica

### Fluxo Híbrido: Diarization + Recognition

```python
class SpeakerIdentifier:
    def __init__(self):
        # Carrega embeddings cadastrados (mesmo database do Verification)
        self.enrolled_embeddings = {
            "user_1": np.load("/data/embeddings/user_1.npy"),
            "user_2": np.load("/data/embeddings/user_2.npy")
        }
        self.threshold = 0.70
    
    async def identify_and_diarize(self, audio, transcript):
        # 1. DIARIZATION: Separa segmentos por voz
        diarization = self.diarize(audio)
        # Output: [(0s-5s, speaker_0), (6s-10s, speaker_1)]
        
        results = []
        for segment in diarization:
            start, end, speaker_cluster = segment
            audio_segment = audio[start:end]
            
            # 2. RECOGNITION: Identifica quem é
            embedding = self.create_embedding(audio_segment)
            
            # Compara com cadastrados
            similarities = {}
            for user_id, enrolled_emb in self.enrolled_embeddings.items():
                sim = cosine_similarity(embedding, enrolled_emb)
                similarities[user_id] = sim
            
            best_match = max(similarities.items(), key=lambda x: x[1])
            user_id, confidence = best_match
            
            # 3. Valida threshold
            if confidence >= self.threshold:
                recognized = True
            else:
                user_id = f"unknown_{hash(embedding)}"
                recognized = False
            
            # 4. Publica resultado
            results.append({
                "text": transcript[start:end],
                "speaker_id": user_id,
                "confidence": confidence,
                "recognized": recognized,
                "start_time": start,
                "end_time": end
            })
        
        return results
```

### Diferença: Diarization Puro vs Híbrido

```python
# ❌ Diarization Puro (anônimo)
Output: [
  {"speaker": "Speaker_0", "text": "qual a temperatura"},  # Quem é?
  {"speaker": "Speaker_1", "text": "desliga o alarme"}     # Quem é?
]
# Problema: Não sabe QUEM é cada speaker!

# ✅ Híbrido (com Recognition)
Output: [
  {"speaker_id": "user_1", "recognized": true, "text": "qual a temperatura"},
  {"speaker_id": "unknown_xyz", "recognized": false, "text": "desliga o alarme"}
]
# Solução: Identifica user_id específico OU unknown
```

---

## 📈 Métricas

```python
speaker_identifications_total{speaker_id}  # user_1, user_2, unknown
speaker_recognition_success_rate           # recognized=true / total
speaker_unknown_detections_total           # recognized=false
speaker_overlap_detections_total
speaker_diarization_latency_seconds
speaker_confidence_avg{speaker_id}
source_separation_triggers_total
```

---

## 🔗 Integração

**Recebe de:** Whisper ASR (gRPC - áudio + texto)  
**Envia para:** Conversation Manager (NATS - speech.diarized)  
**Compartilha:** Embeddings com Speaker Verification (volume `/data/embeddings`)  
**Monitora:** Prometheus, Loki

---

**Versão:** 1.0
