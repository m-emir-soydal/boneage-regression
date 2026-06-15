from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE

def build_multi_input_model(dropout=0.5, learning_rate=1e-4):
    """
    Builds the multi-input model architecture:
    EfficientNetB3 for the image, concatenated with the normalized sex input, 
    followed by a dense prediction head.
    """
    img_input = Input(shape=(*IMG_SIZE, 3), name='image_input')
    sex_input = Input(shape=(1,), name='sex_input')

    # Base model - no top layer
    x = EfficientNetB3(weights='imagenet', include_top=False)(img_input)
    feat = GlobalAveragePooling2D()(x)

    # Concat features and sex
    x = Concatenate()([feat, sex_input])
    
    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    # Regression Output
    out = Dense(1, activation='linear', name='boneage_output')(x)

    # Compile the model
    m = Model(inputs=[img_input, sex_input], outputs=out, name='multi_input')
    m.compile(
        optimizer=Adam(learning_rate),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    
    return m
