from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import torch
import re
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from translate import Translator

loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()


def predict(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


bot = Bot(token="6249096550:AAERkjp9Qnjuu4q6xoK34u3ZDEerjom11GM")
dp = Dispatcher(bot, storage=MemoryStorage())


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer(text=f"Привет👋 {message.from_user.first_name}\nПришли мне ссылку на фото:")


def is_link(text):
    pattern = re.compile(r'(?:http\:|https\:)?\/\/.*\.(?:(png|jpg|jpeg|webp))')
    return pattern.match(text) is not None


@dp.message_handler()
async def get_link(message: types.Message):
    send_message = await message.reply(text=f"Ссылку принял, работаю")
    link = message.text
    if is_link(message.text):
        try:
            with Image.open(requests.get(link, stream=True).raw) as image:
                preds = str(predict(image)[0])
            translator = Translator(from_lang="en", to_lang="ru")
            translation = translator.translate(preds)
            await bot.edit_message_text(text=translation + "\n" + preds, chat_id=message.from_user.id,
                                        message_id=send_message.message_id)
        except Exception as ex:
            await message.answer(text="Ссылка неккоректна. Попоробуйте снова")
    else:
        await bot.edit_message_text(text="Вы прислали не ссылку. Попоробуйте снова", chat_id=message.from_user.id,
                                    message_id=send_message.message_id)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
