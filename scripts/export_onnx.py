"""
Export SpeechBrain ECAPA-TDNN to ONNX format.
Used in multi-stage Docker build — this script runs in the builder stage
and produces ecapa_tdnn.onnx for the runtime stage.
"""
import torch
from speechbrain.inference.speaker import EncoderClassifier

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
CACHE_DIR = "/tmp/speechbrain_cache"
OUTPUT_PATH = "/model/ecapa_tdnn.onnx"


def main():
    print(f"Loading model: {MODEL_SOURCE}")
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=CACHE_DIR,
        run_opts={"device": "cpu"},
    )

    # ECAPA-TDNN embedding model expects (batch, time) waveform
    dummy_input = torch.randn(1, 16000)  # 1 second of 16kHz audio
    dummy_lens = torch.tensor([1.0])     # relative length

    embedding_model = classifier.mods.embedding_model

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, wav, wav_lens):
            return self.model(wav, wav_lens)

    wrapper = ExportWrapper(embedding_model)
    wrapper.eval()

    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_lens),
        OUTPUT_PATH,
        input_names=["wav", "wav_lens"],
        output_names=["embedding"],
        dynamic_axes={
            "wav": {0: "batch", 1: "time"},
            "wav_lens": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
