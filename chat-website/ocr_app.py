import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🧠 Advanced OCR Text Extractor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .result-box {
        background: #e8f4f8;
        border: 2px solid #bee5eb;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧠 Advanced OCR Text Extractor</h1>', unsafe_allow_html=True)

# Try to import EasyOCR
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class AdvancedOCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing for better OCR results"""
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Apply different preprocessing techniques
        processed_images = {}
        
        # 1. Original grayscale
        processed_images['original'] = gray
        
        # 2. Gaussian blur + threshold
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images['gaussian_threshold'] = thresh1
        
        # 3. Median blur + threshold
        median = cv2.medianBlur(gray, 3)
        _, thresh2 = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images['median_threshold'] = thresh2
        
        # 4. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images['adaptive_threshold'] = adaptive
        
        return processed_images
    
    def extract_text_advanced(self, image):
        """Extract text using multiple preprocessing methods"""
        try:
            # Get preprocessed images
            processed_images = self.preprocess_image(image)
            
            best_result = ""
            best_confidence = 0
            best_method = ""
            all_results = []
            
            # Try each preprocessing method
            for method_name, processed_img in processed_images.items():
                # Convert back to PIL for EasyOCR
                if len(processed_img.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(processed_img)
                else:  # Color
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Extract text
                results = self.reader.readtext(img_byte_arr)
                
                # Process results
                extracted_text = ""
                total_confidence = 0
                valid_detections = 0
                
                for bbox, text, confidence in results:
                    if confidence > 0.2:  # Lower threshold to catch more text
                        extracted_text += text + " "
                        total_confidence += confidence
                        valid_detections += 1
                
                if valid_detections > 0:
                    avg_confidence = total_confidence / valid_detections
                    
                    # Store results
                    all_results.append({
                        'method': method_name,
                        'text': extracted_text.strip(),
                        'confidence': avg_confidence,
                        'detections': valid_detections
                    })
                    
                    # Track best result
                    if len(extracted_text) > len(best_result) and avg_confidence > 0.3:
                        best_result = extracted_text.strip()
                        best_confidence = avg_confidence
                        best_method = method_name
            
            return best_result, best_confidence, best_method, all_results
            
        except Exception as e:
            return "", 0, "", []

def get_confidence_color(confidence):
    """Get color based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

# Main application
if not OCR_AVAILABLE:
    st.markdown("""
    <div class="warning-box">
        <h2>❌ EasyOCR Not Installed</h2>
        <p><strong>To enable advanced OCR, install EasyOCR:</strong></p>
        <div style="background: #2d2d2d; color: white; padding: 15px; border-radius: 5px; margin: 15px 0;">
        pip install easyocr opencv-python
        </div>
        <p>After installation, <strong>restart this app</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success("✅ Advanced OCR Engine Ready!")
    
    # Upload section
    st.markdown("---")
    st.subheader("📷 Upload Image for Text Extraction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="For best results, use clear images with good contrast"
        )
    
    with col2:
        extraction_mode = st.selectbox(
            "Extraction Mode",
            ["Standard", "Advanced (Multiple Methods)"],
            help="Advanced mode tries multiple preprocessing techniques"
        )
    
    if uploaded_image is not None:
        # Display the image
        image = Image.open(uploaded_image)
        
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col_info:
            st.info("**Image Information:**")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            
            # Tips for better OCR
            st.info("**💡 Tips for Better OCR:**")
            st.write("• Use high-contrast images")
            st.write("• Ensure text is clear and sharp")
            st.write("• Avoid blurry or distorted images")
            st.write("• Good lighting reduces noise")
        
        # OCR extraction
        if st.button("🚀 Extract Text", type="primary", use_container_width=True):
            processor = AdvancedOCRProcessor()
            
            with st.spinner("🔍 Analyzing image and extracting text..."):
                if extraction_mode == "Standard":
                    # Simple extraction
                    try:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        results = processor.reader.readtext(img_byte_arr)
                        
                        if results:
                            extracted_text = ""
                            confidence_scores = []
                            
                            for i, (bbox, text, confidence) in enumerate(results):
                                if confidence > 0.3:
                                    extracted_text += text + " "
                                    confidence_scores.append(confidence)
                            
                            if extracted_text.strip():
                                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                                
                                st.markdown(f"""
                                <div class="success-box">
                                    <h3>✅ Text Extraction Successful!</h3>
                                    <p>Found {len(results)} text regions • Average confidence: <span class="{get_confidence_color(avg_confidence)}">{avg_confidence:.2f}</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display results
                                col_text, col_stats = st.columns([3, 1])
                                
                                with col_text:
                                    st.subheader("📜 Extracted Text")
                                    st.text_area("Text Output", extracted_text.strip(), height=200, key="extracted_text")
                                
                                with col_stats:
                                    st.subheader("📊 Statistics")
                                    st.metric("Text Regions", len(results))
                                    st.metric("Confident Detections", len(confidence_scores))
                                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Text",
                                    data=extracted_text.strip(),
                                    file_name="extracted_text.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.error("❌ No confident text detected")
                        else:
                            st.error("❌ No text found in the image")
                            
                    except Exception as e:
                        st.error(f"❌ Extraction error: {str(e)}")
                
                else:  # Advanced mode
                    best_text, best_confidence, best_method, all_results = processor.extract_text_advanced(image)
                    
                    if best_text:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Advanced Text Extraction Complete!</h3>
                            <p>Best method: <strong>{best_method}</strong> • Confidence: <span class="{get_confidence_color(best_confidence)}">{best_confidence:.2f}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display best result
                        st.subheader("🏆 Best Extraction Result")
                        st.text_area("Optimized Text Output", best_text, height=150, key="best_text")
                        
                        # Method comparison
                        st.subheader("🔬 Method Comparison")
                        for result in all_results:
                            confidence_class = get_confidence_color(result['confidence'])
                            with st.expander(f"{result['method']} - {result['detections']} detections - Confidence: <span class='{confidence_class}'>{result['confidence']:.2f}</span>", unsafe_allow_html=True):
                                st.text(result['text'] if result['text'] else "No text detected")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Best Text",
                            data=best_text,
                            file_name="optimized_extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ No text could be extracted using any method")
    
    # Instructions
    st.markdown("---")
    st.subheader("🎯 Best Practices for Clear Text Extraction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🖼️ Image Quality**")
        st.write("• High resolution")
        st.write("• Good lighting")
        st.write("• Sharp focus")
    
    with col2:
        st.info("**📝 Text Characteristics**")
        st.write("• Clear fonts")
        st.write("• Good contrast")
        st.write("• Proper spacing")
    
    with col3:
        st.info("**⚙️ Processing**")
        st.write("• Use Advanced mode")
        st.write("• Try different images")
        st.write("• Check confidence scores")

# Footer
st.markdown("---")
st.caption("Powered by EasyOCR • Advanced text extraction with multiple preprocessing methods")