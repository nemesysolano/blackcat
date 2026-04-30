import sys
import tensorflow as tf

def export_model(keras_model_path, litert_model_path):
    print(f"Loading existing Keras model from '{keras_model_path}'...")
    model = tf.keras.models.load_model(keras_model_path)
    
    input_dim = model.input_shape[1]
    
    # --- THE CRITICAL FIX ---
    # Run a dummy pass to force macOS Apple Silicon to load the trained 
    # Batch Normalization weights into memory BEFORE we freeze the graph.
    print("Running dummy forward pass to load weights...")
    dummy_input = tf.zeros((1, input_dim, 1))
    model(dummy_input, training=False)
    # ------------------------
    
    # 1. Lock the model into inference mode. 
    model.trainable = False 
    
    # 2. Wrap the model call in a tf.function
    @tf.function
    def inference_wrapper(x):
        return model(x, training=False)
        
    # 3. Trace the function to create a "Concrete Function".
    print("Tracing graph into a Concrete Function...")
    input_signature = tf.TensorSpec(shape=(1, input_dim, 1), dtype=tf.float32)
    concrete_func = inference_wrapper.get_concrete_function(input_signature)
    
    # 4. Initialize the Converter using the frozen concrete function
    print("Converting frozen graph to LiteRT format...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 5. Perform the conversion
    tflite_model = converter.convert()
    
    # 6. Save the compiled model to disk
    with open(litert_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Model exported to {litert_model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export-model.py <path_to_existing_keras_model> <output.tflite>")
        sys.exit(1)
        
    keras_model_path = sys.argv[1]
    litert_model_path = sys.argv[2]
    export_model(keras_model_path, litert_model_path)