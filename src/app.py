import torch
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
min_pixels = 256*28*28
max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


# def infer(messages, history):
    # Messages containing a images list as a video and a text query
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": [
    #                     "file:///path/to/frame1.jpg",
    #                     "file:///path/to/frame2.jpg",
    #                     "file:///path/to/frame3.jpg",
    #                     "file:///path/to/frame4.jpg",
    #                 ],
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]
    #
    # # Messages containing a local video path and a text query
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "file:///path/to/video1.mp4",
    #                 "max_pixels": 360 * 420,
    #                 "fps": 1.0,
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]
    #
    # # Messages containing a video url and a text query
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]

    # In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     fps=fps,
    #     padding=True,
    #     return_tensors="pt",
    #     **video_kwargs,
    # )
    # inputs = inputs.to("cuda")
    #
    # # Inference
    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    #
    # return


with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")

    @greet_btn.click(inputs=name, outputs=output)
    def greet(name):
        return "Hello " + name + "!"

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')