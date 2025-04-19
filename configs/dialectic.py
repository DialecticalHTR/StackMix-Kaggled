# -*- coding: utf-8 -*-

from .base import BaseConfig


class OldDialecticConfig(BaseConfig):
    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='dialectic',
            chars=' !"%\'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№\u0301\u0302\u0304\u0306\u0311‿jЪЫЬЁ#Il',
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


class DialecticConfig(BaseConfig):
    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='dialectic',
            chars=' !"%\'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№\u0301\u0302\u0304\u0306\u0311‿jЪЫЬЁ#IlV',
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
