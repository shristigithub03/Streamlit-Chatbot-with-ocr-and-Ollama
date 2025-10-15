try:
    import streamlit as st
    from streamlit_chat import message
    print("✅ All imports successful!")
    print(f"Streamlit version: {st.__version__}")
    
    # Test basic functionality
    st.title("Test Page")
    st.success("Everything is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    