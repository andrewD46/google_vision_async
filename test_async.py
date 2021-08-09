import glob
import time
import base64
import asyncio
import aiohttp
import pandas as pd
import cv2
import numpy as np
from pdf2image import pdfinfo_from_path, convert_from_path


GOOGLE_CLOUD_KEY = 'AIzaSyC0yJstNgfMWZgQuhx8xcuYNjswjDTa0YI'
URL = f'https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_KEY}'
WIDTH = 2048
HEIGHT = 2896


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.encodebytes(image_file.read()).decode('utf-8')


def get_tasks(session, imgs_path):
    tasks = []

    func = lambda arr, n=16: [arr[i:i + n] for i in range(0, len(arr), n)]
    imgs_path_batchs = func(imgs_path)

    data_batchs = []
    for imgs_path_batch in imgs_path_batchs:
        data = {
            "requests": [
            ]
        }
        for img_path in imgs_path_batch:
            data['requests'].append(
                {
                    "image": {
                        "content": encode_image(img_path)
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION",
                        }
                    ]
                }
            )
        data_batchs.append(data)
        tasks.append(asyncio.create_task(session.post(URL, json=data)))
    return tasks


async def get_text(imgs_path):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(session, imgs_path)
        responses = await asyncio.gather(*tasks)

        for response in responses:
            results.append(await response.json())
    return results


def get_df(responses):
    df = pd.DataFrame(columns=['description', 'x1', 'y1', 'x2', 'y2'])
    data = {'description': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], }
    for response in responses:
        for page in response['responses']:
            for row in page['textAnnotations']:
                data['description'].append(row['description'])
                x1, y1 = row['boundingPoly']['vertices'][0].values()
                x2, y2 = row['boundingPoly']['vertices'][2].values()
                data['x1'].append(x1)
                data['y1'].append(y1)
                data['x2'].append(x2)
                data['y2'].append(y2)

        dff = pd.DataFrame(data)
        df = df.append(dff)
    return df


def main(pdf_path, save_path):
    imgs_path = sorted(glob.glob(f'{save_path}*'), key=lambda x: int(x.split('/')[2][:-4]))
    data = asyncio.run(get_text(imgs_path))
    df = get_df(data)
    return df


def get_imgs(path_to_pdf, save_path, grayscale=True):
    info = pdfinfo_from_path(path_to_pdf, userpw=None, poppler_path=None)
    try:
        pil_imgs = convert_from_path(path_to_pdf, size=(
            WIDTH, HEIGHT), grayscale=grayscale)
    except MemoryError:
        pil_imgs = []
        maxPages = info['Pages']
        for page in range(1, maxPages + 1, 10):
            pil_imgs.append(convert_from_path(path_to_pdf, size=(
                WIDTH, HEIGHT), grayscale=grayscale, first_page=page, last_page=min(page + 10 - 1, maxPages)))
        pil_imgs = np.concatenate(pil_imgs).flat
    [cv2.imwrite(f'{save_path}{i}.jpg', np.array(img)) for i, img in enumerate(pil_imgs)]
    return True


if __name__ == "__main__":
    imgs_path = 'volume/imgs/'
    pdf_file = 'volume/63 - PWPA Amendment Agreement.pdf'
    get_imgs(pdf_file, imgs_path)
    start = time.time()
    df = main(pdf_file, imgs_path)
    print("Get text: %s seconds ---" % (time.time() - start))
    df.to_excel('out.xlsx')
