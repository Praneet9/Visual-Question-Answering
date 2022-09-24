import gradio as gr
from inference import Inference
import yaml


def predict(image_path, question):
    
    confs, classes = inference.predict_classes(image_path, question)

    predictions = dict()

    for idx, class_name in enumerate(classes):
        predictions[class_name] = float(round(confs[idx], 2))
    
    return predictions


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_path = 'models/ModelRaw_EPOCHS_15_VAL_LOSS_1.12847.pth'
    inference = Inference(config, model_path)
    inference.load_models()

    gr.Interface(fn=predict, 
                description="Ask me questions related to count, color, Yes/No and I'll try my best to answer them!",
                inputs=[gr.Image(label='Image'), gr.Textbox(label='Question')],
                outputs=[gr.components.Label(label='Answer', num_top_classes=config['top_classes'])],
                examples=[['examples/COCO_val2014_000000000776.jpg', 'How many toys are there?'],
                          ['examples/source.jpeg', 'How many leaves are in the picture?'],
                          ['examples/COCO_val2014_000000131089.jpg', 'Is the boy playing baseball?'],
                          ['examples/COCO_val2014_000000262162.jpg', 'Is that a folding chair?'],
                          ['examples/COCO_val2014_000000240301.jpg', 'Is there daylight in the picture?'],
                          ['examples/COCO_val2014_000000002759.jpg', 'How many bowls are there?'],
                          ['examples/COCO_val2014_000000004988.jpg', 'What color is the bus?'],
                          ['examples/COCO_val2014_000000003480.jpg', 'How many cows are there?']]
                ).launch()