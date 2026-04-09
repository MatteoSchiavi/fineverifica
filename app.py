import streamlit as st
import torch
import torch.nn as nn
import string
import time

# --- CONFIGURAZIONE CORE ---
SEQ_LEN = 35
CHARS = string.ascii_letters
VOCAB = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
for i, c in enumerate(CHARS):
    VOCAB[c] = i + 3
INV_VOCAB = {i: c for c, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
DIMENSIONE_EMBEDDING = 128

# --- ARCHITETTURA (Deve essere qui per caricare i pesi) ---
class StringRotatorPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, DIMENSIONE_EMBEDDING)
        self.pos_embedding = nn.Embedding(SEQ_LEN + 5, DIMENSIONE_EMBEDDING)
        self.k_encoder = nn.Linear(1, DIMENSIONE_EMBEDDING)
        self.transformer = nn.Transformer(
            d_model=DIMENSIONE_EMBEDDING, nhead=8, 
            num_encoder_layers=3, num_decoder_layers=3,
            batch_first=True, dropout=0.1
        )
        self.fc_out = nn.Linear(DIMENSIONE_EMBEDDING, VOCAB_SIZE)

    def forward(self, src, tgt, k_val):
        batch_size, src_seq_len = src.size()
        _, tgt_seq_len = tgt.size()
        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        src_emb = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(tgt_pos)
        k_emb = self.k_encoder(k_val).unsqueeze(1)
        
        src_padding_mask = (src == VOCAB['<PAD>'])
        tgt_padding_mask = (tgt == VOCAB['<PAD>'])
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        out = self.transformer(
            src_emb + k_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(out)

# --- INIZIALIZZAZIONE CLOUD ---
st.set_page_config(page_title="Rotazione Neurale", page_icon="🌌", layout="centered")

@st.cache_resource
def load_model():
    model = StringRotatorPro()
    # Mappiamo su CPU perché i server gratuiti di Streamlit non hanno GPU Nvidia
    model.load_state_dict(torch.load("cervello_rotante.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    modello_ai = load_model()
except Exception as e:
    st.error("Rete neurale offline. Hai dimenticato di caricare 'cervello_rotante.pth'?")
    st.stop()

# --- INTERFACCIA ---
st.title("🌌 Rotazione Stringhe con modelli neurali")
st.markdown("Perché scambiare byte in RAM quando puoi usare l'attenzione quantistica di **1.2 milioni di parametri**?")

st.markdown("---")

testo_input = st.text_input("Inserisci stringa (Solo A-Z, a-z):", "Ingegneria")
k_input = st.number_input("Passi di shift temporale (K):", value=4, step=1)

if st.button("Avvia Inferenza", type="primary", use_container_width=True):
    # CRASH VOLUTO 1: Rotazione a sinistra
    if k_input < 0:
        st.error(f"**[CRITICAL FAILURE] Tentativo di inversione entropica rilevato (k={k_input}).**")
        st.warning("La Rete Neurale è addestrata ESCLUSIVAMENTE per il flusso temporale in avanti (Rotazione a Destra). Spostarsi a sinistra richiederebbe il calcolo di antimateria non supportato dai cluster attuali. Il sistema si è arrestato per prevenire un collasso dimensionale.")
        st.stop()
        
    # CRASH VOLUTO 2: Caratteri illegali
    try:
        src_chars = [VOCAB[c] for c in testo_input]
    except KeyError as e:
        st.error(f"**ECCEZIONE SINTATTICA FATALE!** Carattere {e} non riconosciuto dalla matrice di embedding. Il sistema accetta solo lettere pure.")
        st.stop()

    if len(testo_input) > SEQ_LEN - 2:
        st.error(f"Stringa troppo lunga. Massimo {SEQ_LEN - 2} caratteri.")
        st.stop()

    # --- FINTA SUSPENSE ---
    progress_text = "Allocazione di tensori spaziali nella VRAM..."
    my_bar = st.progress(0, text=progress_text)
    
    time.sleep(0.5)
    my_bar.progress(30, text="Riscaldamento della Multi-Head Attention...")
    time.sleep(0.6)
    my_bar.progress(70, text="Calcolo dei gradienti probabilistici in corso...")
    time.sleep(0.7)
    my_bar.progress(100, text="Decodifica dell'output...")
    time.sleep(0.3)
    my_bar.empty()
    
    # --- VERA INFERENZA ---
    start_time = time.time()
    
    src = torch.tensor(src_chars + [VOCAB['<PAD>']] * (SEQ_LEN - len(src_chars)), dtype=torch.long).unsqueeze(0)
    k_tensor = torch.tensor([[k_input / SEQ_LEN]], dtype=torch.float32)
    tgt_chars = [VOCAB['<SOS>']]
    risultato_ai = ""
    
    with torch.no_grad():
        for _ in range(len(testo_input)):
            tgt = torch.tensor(tgt_chars, dtype=torch.long).unsqueeze(0)
            output = modello_ai(src, tgt, k_tensor)
            next_token = output[0, -1, :].argmax().item()
            if next_token == VOCAB['<EOS>'] or next_token == VOCAB['<PAD>']: break
            risultato_ai += INV_VOCAB.get(next_token, '?')
            tgt_chars.append(next_token)
            
    tempo_esecuzione = time.time() - start_time

    # Risultato C per controllo
    k_eff = k_input % len(testo_input) if len(testo_input) > 0 else 0
    risultato_c = testo_input[-k_eff:] + testo_input[:-k_eff] if k_eff > 0 else testo_input

    # --- OUTPUT ---
    st.success("Inferenza completata con successo spaziotemporale.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risultato Rete Neurale", risultato_ai, f"Completato in {tempo_esecuzione:.4f}s")
    with col2:
        st.metric("Risultato in matematico", risultato_c, "O(N) tempo")

    if risultato_ai == risultato_c:
        st.balloons()