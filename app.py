import streamlit as st
import torch
from nougat import NougatModel
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from pathlib import Path
import tempfile
import re

# --- Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="PDF to LaTeX Converter",
    page_icon="🤖",
    layout="wide",
)

# --- Tải và cache mô hình Nougat ---
# Sử dụng st.cache_resource để mô hình chỉ được tải một lần duy nhất
@st.cache_resource
def load_model():
    """Tải mô hình Nougat và cache lại. Lần đầu sẽ mất thời gian."""
    st.write("Tải mô hình Nougat... (Lần đầu có thể mất vài phút)")
    # Lấy đường dẫn checkpoint của mô hình đã huấn luyện
    checkpoint_path = get_checkpoint() 
    # Tải mô hình
    model = NougatModel.from_pretrained(checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

# --- Hàm xử lý chính ---
def convert_pdf_to_latex_like(pdf_file, model):
    """
    Chuyển đổi một tệp PDF thành văn bản giống LaTeX bằng mô hình Nougat.
    """
    try:
        # Nougat cần một đường dẫn file, vì vậy chúng ta lưu file tạm thời
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_pdf_path = Path(tmpdir) / "input.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Tạo dataset từ file PDF
            dataset = ImageDataset(
                pdf_path=temp_pdf_path,
                partial_ocr=False,
                pages=None, # Xử lý tất cả các trang
            )
            
            # Bắt đầu quá trình chuyển đổi
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1, # Xử lý từng trang một
                shuffle=False,
                collate_fn=ImageDataset.collate_fn,
            )

            predictions = []
            for i, (sample, is_last_page) in enumerate(dataloader):
                st.info(f"Đang xử lý trang {i + 1}/{len(dataset)}...")
                model_output = model.inference(image_tensors=sample)
                predictions.extend(model_output["predictions"])
        
        # Nối kết quả từ các trang lại
        full_text = "\n".join(predictions)
        
        # Một vài bước xử lý hậu kỳ đơn giản để làm cho nó giống LaTeX hơn
        # Thay thế các thẻ hình ảnh Markdown bằng cú pháp LaTeX
        full_text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'\\begin{figure}\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{images/\2}\n    \\caption{\1}\n    \\label{fig:\2}\n\\end{figure}', full_text)

        return full_text

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
        return None

# --- Giao diện người dùng Streamlit ---

st.title("🤖 PDF sang LaTeX (Sử dụng Nougat AI)")
st.markdown("""
**LƯU Ý QUAN TRỌNG:**
- Ứng dụng này sử dụng mô hình AI **Nougat** của Meta để chuyển đổi PDF sang định dạng Markdown toán học (rất giống LaTeX).
- **Hoạt động tốt nhất** với các tài liệu khoa học và học thuật (file PDF được tạo từ máy tính, không phải file scan).
- **Quá trình xử lý có thể chậm,** đặc biệt là trong lần chạy đầu tiên vì cần tải mô hình về máy của bạn.
- Kết quả có thể cần một vài chỉnh sửa nhỏ để biên dịch hoàn hảo với trình biên dịch LaTeX của bạn.
""")

# Tải mô hình (sẽ được cache lại)
model = load_model()

# Khu vực tải tệp lên
uploaded_file = st.file_uploader(
    "Chọn một tệp PDF học thuật để chuyển đổi",
    type="pdf",
    help="Kéo và thả hoặc nhấn để tải lên."
)

if uploaded_file is not None:
    st.write("---")
    st.write(f"**Tên tệp:** `{uploaded_file.name}`")
    st.write("---")

    if st.button("Bắt đầu chuyển đổi sang LaTeX", type="primary"):
        with st.spinner("Đang phân tích và chuyển đổi PDF... Quá trình này có thể mất một lúc."):
            latex_output = convert_pdf_to_latex_like(uploaded_file, model)
        
        if latex_output:
            st.success("✅ Chuyển đổi thành công!")
            
            st.subheader("Kết quả (Định dạng Markdown/LaTeX)")
            st.code(latex_output, language='latex')

            # Nút tải xuống
            download_filename = f"{uploaded_file.name.rsplit('.', 1)[0]}.tex"
            st.download_button(
                label="📥 Tải xuống file .tex",
                data=latex_output.encode('utf-8'),
                file_name=download_filename,
                mime='text/x-tex'
            )
        else:
            st.error("Không thể chuyển đổi tệp PDF. Vui lòng thử một tệp khác.")

# --- Chân trang ---
st.markdown("---")
st.markdown("Xây dựng với ❤️ bởi [Streamlit](https://streamlit.io) & [Nougat AI](https://github.com/facebookresearch/nougat).")
