# -*- coding: utf-8 -*-

from .base import BaseConfig

class CyrillicConfig(BaseConfig):
    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='bentham',
            chars=' !"%\'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№',
            corpus_name='jigsaw_corpus.txt',
            blank='ß',
            **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            image_w=image_w,
            image_h=image_h,
            chars=chars,
            blank=blank,
            corpus_name=corpus_name,
            **kwargs,
        )
