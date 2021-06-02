import os, logging, shutil, argparse, img2pdf, PIL, glob, re
import lxml.html
import asyncio, aiohttp, aiofiles
from collections import deque
import numpy as np
import multiprocessing
from typing import List, Optional
import io
import cfscrape

logging.basicConfig(level=logging.INFO)


def get_next_chapter(root):
    link = root.xpath(".//div[@class='nav-links']//a//@href")
    return link


async def download_img(session:aiohttp.ClientSession, 
                                url:str, dest:str)->None:

    async with session.get(url) as resp:
        bytes = await resp.read()
    async with aiofiles.open(dest, "wb") as f:
        await f.write(bytes)
    logging.info(f"Downloaded {url}")


async def download_chapter(session:aiohttp.ClientSession, 
                            root:lxml.etree, dest:str) -> None:
    logging.info("\n")
    os.makedirs(dest, exist_ok=True)
    pages_root = root.xpath(".//div[@class='reading-content']")[0]
    imgs = pages_root.xpath(".//img/@src")
    coros = []
    for i,img in enumerate(imgs):
        img = img.strip()
        dest_ = f"{dest}/{i+1}.jpeg"
        coros.append(download_img(session, img, dest_))
        # await asyncio.create_task(download_img(session, img, dest_)) 
    await asyncio.gather(*coros)
    logging.info(f"Downloaded {os.path.basename(dest)}")



async def download_series(url:str, save_path:str, 
                                n_chapters:Optional[int] = None) -> None: 
    pages= deque()
    next_chapter = url
    count = 0
    n_chapters_init = n_chapters
    if n_chapters == None:
        n_chapters = float('inf')
    async with aiohttp.ClientSession() as session:

        while next_chapter != "#" and count < n_chapters:
            logging.info(f'requesting   {next_chapter}')
            async with session.get(next_chapter) as resp:
                resp = await resp.text()
            name = next_chapter.split("/")[-2]
            root = lxml.html.fromstring(resp)
            next_chapter = get_next_chapter(root)[1] 
            pages.append([name,root])
            count += 1
        if n_chapters_init == None:
            pages.pop()
        coros = []
        while pages:
            name, resp = pages.popleft()
            dest = os.path.join(save_path, name)
            coros.append(download_chapter(session, resp, dest)) 
            await asyncio.create_task(download_chapter(session, resp, dest))



async def download_series_cfscrape(url:str, save_path:str, 
                                n_chapters:Optional[int] = None) -> None: 
    pages= deque()
    next_chapter = url
    count = 0
    n_chapters_init = n_chapters
    if n_chapters == None:
        n_chapters = float('inf')
    
    scraper = cfscrape.create_scraper()  

    while next_chapter != "#" and count < n_chapters:
        logging.info(f'requesting   {next_chapter}')
        resp = scraper.get(next_chapter).content  
        name = next_chapter.split("/")[-2]
        root = lxml.html.fromstring(resp)
        next_chapter = get_next_chapter(root)[1] 
        pages.append([name,root])
        count += 1
    if n_chapters_init == None:
        pages.pop()
      
    while pages:
        name, resp = pages.popleft()
        dest = os.path.join(save_path, name)
        async with aiohttp.ClientSession() as session:
            await asyncio.create_task(download_chapter(session, resp, dest))


def pad_images(img_arrs: List[np.ndarray], pad_width:int) -> np.ndarray:
    padded_img_arrs = []
    for img in img_arrs:
        h,w = img.shape[:2]
        to_pad = pad_width - w
        if to_pad%2 == 0:
            pad = np.full(shape=(h, to_pad//2, 3),fill_value=255).astype(np.uint8)
            img = np.hstack((pad, img, pad))
        else:
            left = np.full(shape=(h, (to_pad//2)+1, 3),fill_value=255).astype(np.uint8)
            right = np.full(shape=(h, (to_pad//2), 3),fill_value=255).astype(np.uint8)
            img = np.hstack((left, img, right))
        padded_img_arrs.append(img)  
    return np.vstack(padded_img_arrs)


def img_to_pdf(img_path:str, pdf_name:str) -> None:
    with open(pdf_name,'wb') as f:
        f.write(img2pdf.convert(img_path))


def prepare_stritched_chapter(chapter_folder:str, save_path:str,
                                chapter_name:Optional[str] = None) -> None:
    os.makedirs(save_path, exist_ok=True)
    imgs = [i.path for i in os.scandir(chapter_folder) if i.path.endswith(".jpeg")]
    imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    max_width = 0
    img_arrs = []

    for img in imgs:
        img = PIL.Image.open(img)
        img = np.asarray(img)
        _, w = img.shape[:2]
        if max_width < w:
            max_width = w
        img_arrs.append(img)
    
    final_img = pad_images(img_arrs, max_width)
    
    img = PIL.Image.fromarray(final_img)
    filename = os.path.basename(chapter_folder)
    img_save_path = os.path.join(chapter_folder, filename+'.png')
    img.save(img_save_path)

    pdf_save_path = os.path.join(save_path, filename+'.pdf')
    img_to_pdf(img_save_path, pdf_save_path)
    logging.info(f"{filename} finished")


def prepare_lossy_chapter(chapter_folder:str, save_path:str,
                            chapter_name:Optional[str] = None) -> None:

    os.makedirs(save_path, exist_ok=True)
    imgs = [i.path for i in os.scandir(chapter_folder) if i.path.endswith(".jpeg")]
    imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
    if chapter_name is None:
        chapter_name = os.path.basename(chapter_folder)
    pdf_save_path = os.path.join(save_path, chapter_name+'.pdf')
    img_to_pdf(imgs, pdf_save_path)
    logging.info(f"{chapter_name} finished")


def prepare_hd_chapter(chapter_folder:str, save_path:str,
                        chapter_name:Optional[str] = None) -> None:

    os.makedirs(save_path, exist_ok=True)
    imgs = [i.path for i in os.scandir(chapter_folder) if i.path.endswith(".jpeg")]
    imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
    if chapter_name is None:
        chapter_name = os.path.basename(chapter_folder)
    pdf_save_path = os.path.join(save_path, chapter_name+'.pdf')

    png_img_bytes = []
    for img in imgs:
        img = PIL.Image.open(img)
        with io.BytesIO() as f:
            img.save(f, format="PNG")
            png_img_bytes.append(f.getvalue())

    img_to_pdf(png_img_bytes, pdf_save_path)
    logging.info(f"{chapter_name} finished")

    
    


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--series_name', type=str)
    parser.add_argument('--nchapters', type=int)
    return parser

# def validate_arguments(args):
#     if not args.url.startswith('http'):
#         raise("Invalid URL")
#     if args.nchapters <= 0:
#         args.nchapters = None
#     if args.series_name==""



def main():
    parser = parse_arguments()
    args = parser.parse_args()
    url = args.url
    save_path = args.save_path
    if args.nchapters:
        nchapters = args.nchapters
    else:
        nchapters = None
    series_name = args.series_name


    save_path = os.path.join(save_path, series_name)
    os.makedirs(save_path, exist_ok=True)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(download_series(url, save_path, n_chapters=nchapters))
    except IndexError:
        logging.info("Cloudfare blockade")
        asyncio.run(download_series_cfscrape(url, save_path, n_chapters=nchapters))


    PIL.Image.MAX_IMAGE_PIXELS = None

    chapters = [(chapter_folder.path, save_path) \
                    for chapter_folder in os.scandir(save_path) \
                    if os.path.isdir(chapter_folder)]  
    with multiprocessing.Pool(5) as p:
        p.starmap(prepare_hd_chapter, chapters)
    
    
    
    #cleanup
    # for object in os.scandir(save_path):
    #     if os.path.isdir(object):
    #         shutil.rmtree(object)


if __name__ == '__main__':
    main()







   
