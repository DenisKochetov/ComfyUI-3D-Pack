import torch
import os
import trimesh
import numpy as np
from collections import OrderedDict
from huggingface_hub import snapshot_download

import json

from .shared_utils.mesh_utils import Mesh
from .shared_utils.image_utils import torch_imgs_to_pils, pils_to_torch_imgs, pil_make_image_grid

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CKPT_ROOT_PATH = os.path.join(ROOT_PATH, "Checkpoints")
CKPT_DIFFUSERS_PATH = os.path.join(CKPT_ROOT_PATH, "Diffusers")


WEIGHT_DTYPE = torch.float16
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

HF_DOWNLOAD_IGNORE = ["*.yaml", "*.json", "*.py", ".png", ".jpg", ".gif"]



#--- HUN gp START ---
from Hunyuan3D_2GP.hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    Hunyuan3DDiTPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
    MeshSimplifier
)
from Hunyuan3D_2GP.hy3dgen.shapegen.pipelines import export_to_trimesh
from Hunyuan3D_2GP.hy3dgen.texgen import Hunyuan3DPaintPipeline
from Hunyuan3D_2GP.hy3dgen.rembg import BackgroundRemover
from Hunyuan3D_2GP.hy3dgen.text2image import HunyuanDiTPipeline
HUNYUAN3D_AVAILABLE = True

#--- HUN START ---
class Load_Hunyuan3D_ShapeGen_Pipeline:
    """Загрузчик пайплайна Hunyuan3D для генерации формы"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("HUNYUAN3D_PIPE",)
    RETURN_NAMES = ("hunyuan3d_pipe",)
    FUNCTION = "load"

    _REPO_ID_BASE = "tencent"

    _MODES = {
        "Hunyuan3D-2-Turbo":       ("Hunyuan3D-2",     "hunyuan3d-dit-v2-0-turbo",    5),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (list(cls._MODES.keys()), {"default": "Hunyuan3D-2-Turbo"}),
                "weights_format": (["safetensors", "ckpt"], {"default": "safetensors"}),
                "flash_vdm": ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod
    def _ensure_weights(repo: str, subfolder: str, use_safetensors: bool):
        base_dir = os.path.join(CKPT_DIFFUSERS_PATH, f"{Load_Hunyuan3D_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}")
        
        # Определяем какой файл искать в зависимости от формата
        if use_safetensors:
            ckpt_file = "model.fp16.safetensors"
        else:
            ckpt_file = "model.fp16.ckpt"
            
        ckpt_path = os.path.join(base_dir, subfolder, ckpt_file)

        if not os.path.exists(ckpt_path):
            print(f"Скачиваем модель: {Load_Hunyuan3D_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}")
            print(f"Скачиваем только директорию: {subfolder}")
            
            # Определяем паттерны для скачивания в зависимости от формата весов
            patterns_to_download = [f"{subfolder}/**"]
            
            snapshot_download(
                repo_id=f"{Load_Hunyuan3D_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}",
                repo_type="model",
                local_dir=base_dir,
                resume_download=True,
                allow_patterns=patterns_to_download,  # Скачиваем только нужную директорию
                ignore_patterns=HF_DOWNLOAD_IGNORE    # Игнорируем ненужные файлы
            )
            
            # Проверяем что файл действительно скачался
            if not os.path.exists(ckpt_path):
                raise RuntimeError(f"Файл {ckpt_file} не найден в {subfolder} после скачивания")

        return ckpt_path

    @staticmethod
    def _build_pipe(repo: str, subfolder: str, use_safetensors: bool, flash_vdm: bool):
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")

        Load_Hunyuan3D_ShapeGen_Pipeline._ensure_weights(repo, subfolder, use_safetensors)

        model_dir = os.path.join(CKPT_DIFFUSERS_PATH,
                                 f"{Load_Hunyuan3D_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}",
                                 subfolder)
        ckpt = os.path.join(model_dir, "model.fp16.safetensors" if use_safetensors else "model.fp16.ckpt")
        cfg = os.path.join(model_dir, "config.yaml")

        print(f"Загружаем пайплайн из:")
        print(f"  Модель: {ckpt}")
        print(f"  Конфиг: {cfg}")

        try:
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                ckpt_path=ckpt,
                config_path=cfg,
                device="cuda",
                dtype=torch.float16,
                use_safetensors=use_safetensors,
                from_pretrained_kwargs={
                    "model_path": f"{Load_Hunyuan3D_ShapeGen_Pipeline._REPO_ID_BASE}/{repo}",
                    "subfolder": subfolder,
                    "use_safetensors": use_safetensors,
                },
            )
            
            if pipe is None:
                raise RuntimeError("from_single_file вернул None")
            
            print(f"Пайплайн создан успешно. Тип: {type(pipe)}")

            if flash_vdm and any(tag in subfolder for tag in ("turbo", "fast")):
                print("Включаем FlashVDM...")
                try:
                    pipe.enable_flashvdm(replace_vae=False)
                    print("FlashVDM включен успешно")
                except Exception as flash_error:
                    print(f"Ошибка при включении FlashVDM: {flash_error}")
                    print("Продолжаем без FlashVDM")

            # Не присваиваем результат .to() так как он может вернуть None
            pipe.to("cuda", torch.float16)
            print(f"Пайплайн перенесен на CUDA. Тип: {type(pipe)}")
            return pipe
            
        except Exception as e:
            print(f"Ошибка в _build_pipe: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def load(self, generation_mode, weights_format, flash_vdm):
        repo, subfolder, def_steps = self._MODES[generation_mode]
        use_safe = (weights_format == "safetensors")
        
        print(f"Загрузка пайплайна:")
        print(f"  Режим: {generation_mode}")
        print(f"  Репозиторий: {repo}")
        print(f"  Подпапка: {subfolder}")
        print(f"  Шаги по умолчанию: {def_steps}")
        print(f"  Использовать safetensors: {use_safe}")
        print(f"  FlashVDM: {flash_vdm}")
        
        try:
            pipe = self._build_pipe(repo, subfolder, use_safe, flash_vdm)
            
            if pipe is None:
                raise RuntimeError("_build_pipe вернул None")
            
            pipe.num_inference_steps = def_steps
            print(f"Hunyuan3D пайплайн загружен: {generation_mode}")
            return (pipe,)
            
        except Exception as e:
            print(f"Ошибка при загрузке пайплайна: {e}")
            import traceback
            traceback.print_exc()
            raise e


class Load_Hunyuan3D_TexGen_Pipeline:
    """Загрузчик пайплайна Hunyuan3D для генерации текстур"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("HUNYUAN3D_TEXGEN_PIPE",)
    RETURN_NAMES = ("hunyuan3d_texgen_pipe",)
    FUNCTION = "load"

    MODEL2REPO = {
        "Turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (list(cls.MODEL2REPO.keys()), {"default": "Turbo"}),
                "enable_cpu_offload": ("BOOLEAN", {"default": False}),
            }
        }

    def _download_required_weights(self, repo_id, subfolder):
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D не доступен. Установите необходимые зависимости.")

        ckpt_download_dir = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)
        
        # Определяем какие директории нужны для TexGen
        required_dirs = ["hunyuan3d-delight-v2-0", subfolder]
        
        # Проверяем что все директории существуют
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = os.path.join(ckpt_download_dir, dir_name)
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            try:
                # Определяем паттерны для скачивания только нужных директорий
                patterns_to_download = []
                for dir_name in missing_dirs:
                    patterns_to_download.append(f"{dir_name}/**")
                
                print(f"Скачиваем недостающие директории: {missing_dirs}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=ckpt_download_dir,
                    repo_type="model",
                    force_download=False,
                    resume_download=True,
                    allow_patterns=patterns_to_download,  # Скачиваем только нужные директории
                    ignore_patterns=HF_DOWNLOAD_IGNORE    # Игнорируем ненужные файлы
                )
                
                print(f"Загрузка завершена для директорий: {missing_dirs}")
                
            except Exception as e:
                print(f"Ошибка при скачивании моделей: {e}")
                raise RuntimeError(f"Не удалось загрузить модели Hunyuan3D TexGen: {e}")
        
        # Проверяем что все ключевые компоненты существуют
        for dir_name in required_dirs:
            dir_path = os.path.join(ckpt_download_dir, dir_name)
            if not os.path.exists(dir_path):
                raise RuntimeError(f"Директория {dir_name} не найдена после скачивания")
                
        return ckpt_download_dir

    def load(self, generation_mode, enable_cpu_offload):
        repo_id, subfolder = self.MODEL2REPO[generation_mode]

        self._download_required_weights(repo_id, subfolder)

        local_repo_dir = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)

        print(f"Загружаем TexGen пайплайн:")
        print(f"  Режим: {generation_mode}")
        print(f"  Репозиторий: {repo_id}")
        print(f"  Подпапка: {subfolder}")
        print(f"  Локальная директория: {local_repo_dir}")

        # Проверяем что все нужные файлы на месте перед загрузкой
        delight_dir = os.path.join(local_repo_dir, "hunyuan3d-delight-v2-0")
        paint_dir = os.path.join(local_repo_dir, subfolder)
        
        if not os.path.exists(delight_dir):
            raise RuntimeError(f"Директория delight не найдена: {delight_dir}")
        if not os.path.exists(paint_dir):
            raise RuntimeError(f"Директория paint не найдена: {paint_dir}")
            
        # Проверяем наличие text_encoder
        text_encoder_dir = os.path.join(delight_dir, "text_encoder")
        if not os.path.exists(text_encoder_dir):
            raise RuntimeError(f"text_encoder не найден: {text_encoder_dir}")

        try:
            # Создаем пайплайн с правильными параметрами
            pipe = Hunyuan3DPaintPipeline.from_pretrained(
                model_path=local_repo_dir
            )

            # Включаем CPU offload если нужно
            if enable_cpu_offload:
                pipe.enable_model_cpu_offload(device="cuda")

            print("Hunyuan3D TexGen пайплайн успешно загружен")
            return (pipe,)
        
        except Exception as e:
            print(f"Ошибка при создании пайплайна: {e}")
            print("Попробуйте удалить папку с моделью и скачать заново:")
            print(f"rm -rf {local_repo_dir}")
            raise e


class Load_Hunyuan3D_Text2Image_Pipeline:
    """Загрузчик пайплайна Hunyuan3D для генерации изображений из текста"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("HUNYUAN3D_T2I_PIPE",)
    RETURN_NAMES = ("hunyuan3d_t2i_pipe",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def load(cls, model_path, device, compile_model):
        
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        # Определяем устройство
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Загрузка Hunyuan3D Text2Image пайплайна:")
        print(f"  Модель: {model_path}")
        print(f"  Устройство: {device}")
        
        pipe = HunyuanDiTPipeline(model_path=model_path, device=device)
        
        if compile_model and device == "cuda":
            pipe.compile()
        
        print("Hunyuan3D Text2Image пайплайн успешно загружен")
        return (pipe,)


class Hunyuan3D_BackgroundRemover:
    """Удаление фона с изображения с помощью Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_no_bg",)
    FUNCTION = "remove_background"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def remove_background(self, image):
        
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        # Преобразование torch tensor в PIL изображение
        if isinstance(image, torch.Tensor):
            pil_images = torch_imgs_to_pils(image)
            input_image = pil_images[0]
        else:
            input_image = image
        
        # Инициализация инструмента удаления фона
        rembg = BackgroundRemover()
        
        # Удаление фона
        output_image = rembg(input_image.convert('RGB'))
        
        # Преобразование обратно в torch tensor
        result_tensor = pils_to_torch_imgs([output_image], device=DEVICE_STR)
        
        return (result_tensor,)


class Hunyuan3D_Text_To_Image:
    """Генерация изображения из текста с помощью Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_t2i_pipe": ("HUNYUAN3D_T2I_PIPE",),
                "prompt": ("STRING", {"default": "красивая девушка в белой студии, 3D стиль, высокое качество", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def generate(self, hunyuan3d_t2i_pipe, prompt, seed):
        
        print(f"Генерация изображения из текста:")
        print(f"  Промпт: {prompt}")
        print(f"  Сид: {seed}")
        
        # Генерация изображения
        generated_image = hunyuan3d_t2i_pipe(prompt, seed=seed)
        
        # Преобразование в torch tensor
        result_tensor = pils_to_torch_imgs([generated_image], device=DEVICE_STR)
        
        return (result_tensor,)


class Hunyuan3D_Image_To_Shape:
    """Генерация 3D формы из изображения с помощью Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path") 
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_pipe": ("HUNYUAN3D_PIPE",),
                "image": (["IMAGE", "LIST"],),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "num_chunks": ("INT", {"default": 8000, "min": 1000, "max": 50000}),
                "box_v": ("FLOAT", {"default": 1.01, "min": 0.5, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "./output/hunyuan3d_shape.glb"}),
                "mc_algo": ("STRING", {"default": "dmc"}),
                "dual_guidance": ("BOOLEAN", {"default": True}),
                "dual_guidance_scale": ("FLOAT", {"default": 10.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    def generate(self, hunyuan3d_pipe, image, num_inference_steps, guidance_scale, 
                 octree_resolution, seed, num_chunks, box_v, save_path="./output/hunyuan3d_shape.glb",
                 mc_algo="dmc", dual_guidance=True, dual_guidance_scale=10.5):
        
        # Преобразование torch tensor в PIL изображение или получение первого элемента из списка
        if isinstance(image, list):
            # Если это список PIL изображений, берем первое
            input_image = image[0]
        elif isinstance(image, torch.Tensor):
            pil_images = torch_imgs_to_pils(image)
            input_image = pil_images[0]
        else:
            input_image = image
        
        # Получение устройства из пайплайна
        pipe_device = DEVICE_STR  # Используем глобальное устройство по умолчанию
        if hasattr(hunyuan3d_pipe, 'device'):
            pipe_device = hunyuan3d_pipe.device
        elif hasattr(hunyuan3d_pipe, 'model') and hasattr(hunyuan3d_pipe.model, 'device'):
            pipe_device = hunyuan3d_pipe.model.device
        
        # Установка seed если указан
        if seed != -1:
            generator = torch.Generator(device=pipe_device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        print(f"Генерация 3D формы из изображения:")
        print(f"  Шаги инференса: {num_inference_steps}")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Разрешение октодерева: {octree_resolution}")
        print(f"  Сид: {seed}")
        print(f"  Устройство: {pipe_device}")
        
        # Генерация 3D меша
        if hasattr(hunyuan3d_pipe, '__call__'):
            # Для FlowMatching пайплайна
            mesh_outputs = hunyuan3d_pipe(
                image=input_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                box_v=box_v,
                mc_algo=mc_algo,
                output_type='mesh'
            )
        else:
            raise ValueError("Неподдерживаемый тип пайплайна")
        
        # Преобразование в trimesh
        if isinstance(mesh_outputs, list):
            trimesh_mesh = export_to_trimesh(mesh_outputs)[0]
        else:
            trimesh_mesh = export_to_trimesh([mesh_outputs])[0]
        
        # Сохранение меша
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trimesh_mesh.export(save_path)
        
        # Получаем абсолютный путь
        absolute_save_path = os.path.abspath(save_path)
        
        # Создание объекта Mesh для ComfyUI
        mesh_obj = Mesh(
            v=trimesh_mesh.vertices,
            f=trimesh_mesh.faces,
            device=DEVICE_STR
        )
        
        print(f"3D форма сгенерирована и сохранена: {absolute_save_path}")
        print(f"  Вершины: {len(trimesh_mesh.vertices)}")
        print(f"  Грани: {len(trimesh_mesh.faces)}")
        
        return (mesh_obj, absolute_save_path)


class Hunyuan3D_Mesh_To_Texture:
    """Генерация текстуры для 3D меша с помощью Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("textured_mesh_path",)
    FUNCTION = "generate_texture"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_texgen_pipe": ("HUNYUAN3D_TEXGEN_PIPE",),
                "mesh": ("MESH",),
                "reference_image": (["IMAGE", "LIST"],),
                "save_path": ("STRING", {"default": "./output/hunyuan3d_textured.glb"}),
            }
        }

    def generate_texture(self, hunyuan3d_texgen_pipe, mesh, reference_image, save_path):
        
        # Преобразование изображения
        if isinstance(reference_image, list):
            # Если это список PIL изображений, берем первое
            ref_image = reference_image[0]
        elif isinstance(reference_image, torch.Tensor):
            pil_images = torch_imgs_to_pils(reference_image)
            ref_image = pil_images[0]
        else:
            ref_image = reference_image
        
        # Преобразование меша в trimesh
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            # Mesh объект ComfyUI
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.v.cpu().numpy() if torch.is_tensor(mesh.v) else mesh.v,
                faces=mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else mesh.f
            )
        else:
            trimesh_mesh = mesh
        
        print(f"Генерация текстуры для 3D меша:")
        print(f"  Вершины: {len(trimesh_mesh.vertices)}")
        print(f"  Грани: {len(trimesh_mesh.faces)}")
        
        # Генерация текстуры
        textured_mesh = hunyuan3d_texgen_pipe(trimesh_mesh, ref_image)
        
        # Сохранение текстурированного меша
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        textured_mesh.export(save_path)
        
        # Получаем абсолютный путь
        absolute_save_path = os.path.abspath(save_path)
        
        print(f"Текстурированный меш сохранен: {absolute_save_path}")
        
        return (absolute_save_path,)


class Hunyuan3D_Multiview_Image_To_Shape:
    """Генерация 3D формы из нескольких видов изображения с помощью Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_pipe": ("HUNYUAN3D_PIPE",),
                "front_image": ("IMAGE",),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "num_chunks": ("INT", {"default": 8000, "min": 1000, "max": 50000}),
                "box_v": ("FLOAT", {"default": 1.01, "min": 0.5, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "save_path": ("STRING", {"default": "./output/hunyuan3d_multiview_shape.glb"}),
                "mc_algo": ("STRING", {"default": "dmc"}),
            }
        }

    def generate(self, hunyuan3d_pipe, front_image, num_inference_steps, guidance_scale,
                 octree_resolution, seed, num_chunks, box_v, back_image=None, left_image=None,
                 right_image=None, save_path="./output/hunyuan3d_multiview_shape.glb", mc_algo="dmc"):
        
        # Подготовка изображений для multiview
        images_dict = {}
        
        # Обязательное переднее изображение
        if isinstance(front_image, torch.Tensor):
            pil_images = torch_imgs_to_pils(front_image)
            images_dict['front'] = pil_images[0]
        else:
            images_dict['front'] = front_image
        
        # Опциональные изображения
        for key, img in [('back', back_image), ('left', left_image), ('right', right_image)]:
            if img is not None:
                if isinstance(img, torch.Tensor):
                    pil_images = torch_imgs_to_pils(img)
                    images_dict[key] = pil_images[0]
                else:
                    images_dict[key] = img
        
        # Получение устройства из пайплайна
        pipe_device = DEVICE_STR  # Используем глобальное устройство по умолчанию
        if hasattr(hunyuan3d_pipe, 'device'):
            pipe_device = hunyuan3d_pipe.device
        elif hasattr(hunyuan3d_pipe, 'model') and hasattr(hunyuan3d_pipe.model, 'device'):
            pipe_device = hunyuan3d_pipe.model.device
        
        # Установка seed если указан
        if seed != -1:
            generator = torch.Generator(device=pipe_device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        print(f"Генерация 3D формы из multiview изображений:")
        print(f"  Доступные виды: {list(images_dict.keys())}")
        print(f"  Шаги инференса: {num_inference_steps}")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Устройство: {pipe_device}")
        
        # Генерация 3D меша
        mesh_outputs = hunyuan3d_pipe(
            image=images_dict,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            box_v=box_v,
            mc_algo=mc_algo,
            output_type='mesh'
        )
        
        # Преобразование в trimesh
        if isinstance(mesh_outputs, list):
            trimesh_mesh = export_to_trimesh(mesh_outputs)[0]
        else:
            trimesh_mesh = export_to_trimesh([mesh_outputs])[0]
        
        # Сохранение меша
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trimesh_mesh.export(save_path)
        
        # Получаем абсолютный путь
        absolute_save_path = os.path.abspath(save_path)
        
        # Создание объекта Mesh для ComfyUI
        mesh_obj = Mesh(
            v=trimesh_mesh.vertices,
            f=trimesh_mesh.faces,
            device=DEVICE_STR
        )
        
        print(f"3D форма сгенерирована и сохранена: {absolute_save_path}")
        print(f"  Вершины: {len(trimesh_mesh.vertices)}")
        print(f"  Грани: {len(trimesh_mesh.faces)}")
        
        return (mesh_obj, absolute_save_path)


class Hunyuan3D_Mesh_Postprocessor:
    """Постобработка 3D меша с помощью инструментов Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("processed_mesh", "processed_mesh_path")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": False}),
                "simplify_mesh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_face_count": ("INT", {"default": 40000, "min": 1000, "max": 200000}),
                "save_path": ("STRING", {"default": "./output/hunyuan3d_processed.glb"}),
            }
        }

    def process(self, mesh, remove_floaters, remove_degenerate_faces, reduce_faces, 
                simplify_mesh, max_face_count=40000, save_path="./output/hunyuan3d_processed.glb"):
        
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        # Преобразование меша в trimesh
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            # Mesh объект ComfyUI
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.v.cpu().numpy() if torch.is_tensor(mesh.v) else mesh.v,
                faces=mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else mesh.f
            )
        else:
            trimesh_mesh = mesh
        
        print(f"Постобработка 3D меша:")
        print(f"  Начальные вершины: {len(trimesh_mesh.vertices)}")
        print(f"  Начальные грани: {len(trimesh_mesh.faces)}")
        
        # Применение постобработчиков
        if remove_floaters:
            floater_remover = FloaterRemover()
            trimesh_mesh = floater_remover(trimesh_mesh)
            print("  Удалены изолированные компоненты")
        
        if remove_degenerate_faces:
            degenerate_remover = DegenerateFaceRemover()
            trimesh_mesh = degenerate_remover(trimesh_mesh)
            print("  Удалены вырожденные грани")
        
        if reduce_faces:
            face_reducer = FaceReducer()
            trimesh_mesh = face_reducer(trimesh_mesh, max_facenum=max_face_count)
            print(f"  Уменьшено количество граней до {max_face_count}")
        
        if simplify_mesh:
            mesh_simplifier = MeshSimplifier()
            trimesh_mesh = mesh_simplifier(trimesh_mesh)
            print("  Применено упрощение меша")
        
        # Сохранение обработанного меша
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trimesh_mesh.export(save_path)
        
        # Получаем абсолютный путь
        absolute_save_path = os.path.abspath(save_path)
        
        # Создание объекта Mesh для ComfyUI
        processed_mesh_obj = Mesh(
            v=torch.from_numpy(trimesh_mesh.vertices).float().to(DEVICE_STR),
            f=torch.from_numpy(trimesh_mesh.faces).long().to(DEVICE_STR),
            device=DEVICE_STR
        )
        
        print(f"Обработанный меш сохранен: {absolute_save_path}")
        print(f"  Финальные вершины: {len(trimesh_mesh.vertices)}")
        print(f"  Финальные грани: {len(trimesh_mesh.faces)}")
        
        return (processed_mesh_obj, absolute_save_path)


class Hunyuan3D_Complete_Pipeline:
    """Полный пайплайн Hunyuan3D: от изображения до текстурированного меша"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("MESH", "STRING", "STRING")
    RETURN_NAMES = ("mesh", "shape_path", "textured_path")
    FUNCTION = "generate_complete"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_pipe": ("HUNYUAN3D_PIPE",),
                "hunyuan3d_texgen_pipe": ("HUNYUAN3D_TEXGEN_PIPE",),
                "image": (["IMAGE", "LIST"],),
                "remove_background": ("BOOLEAN", {"default": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "postprocess": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "save_dir": ("STRING", {"default": "./output"}),
                "filename_base": ("STRING", {"default": "hunyuan3d_complete"}),
            }
        }

    def generate_complete(self, hunyuan3d_pipe, hunyuan3d_texgen_pipe, image, remove_background,
                         num_inference_steps, guidance_scale, octree_resolution, seed, postprocess,
                         save_dir="./output", filename_base="hunyuan3d_complete"):
        
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        # Преобразование torch tensor в PIL изображение или получение первого элемента из списка
        if isinstance(image, list):
            # Если это список PIL изображений, берем первое
            input_image = image[0]
        elif isinstance(image, torch.Tensor):
            pil_images = torch_imgs_to_pils(image)
            input_image = pil_images[0]
        else:
            input_image = image
        
        # Удаление фона если необходимо
        if remove_background:
            rembg = BackgroundRemover()
            if input_image.mode == 'RGB':
                input_image = rembg(input_image)
            print("Фон удален")
        
        # Получение устройства из пайплайна
        pipe_device = DEVICE_STR  # Используем глобальное устройство по умолчанию
        if hasattr(hunyuan3d_pipe, 'device'):
            pipe_device = hunyuan3d_pipe.device
        elif hasattr(hunyuan3d_pipe, 'model') and hasattr(hunyuan3d_pipe.model, 'device'):
            pipe_device = hunyuan3d_pipe.model.device
        
        # Установка seed если указан
        if seed != -1:
            generator = torch.Generator(device=pipe_device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        print(f"Запуск полного пайплайна Hunyuan3D:")
        print(f"  Удаление фона: {remove_background}")
        print(f"  Постобработка: {postprocess}")
        print(f"  Устройство: {pipe_device}")
        
        # Генерация 3D формы
        print("Этап 1: Генерация 3D формы...")
        mesh_outputs = hunyuan3d_pipe(
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            output_type='mesh'
        )
        
        # Преобразование в trimesh
        if isinstance(mesh_outputs, list):
            trimesh_mesh = export_to_trimesh(mesh_outputs)[0]
        else:
            trimesh_mesh = export_to_trimesh([mesh_outputs])[0]
        
        # Постобработка если включена
        if postprocess:
            print("Этап 2: Постобработка меша...")
            face_reducer = FaceReducer()
            trimesh_mesh = face_reducer(trimesh_mesh)
        
        # Сохранение формы
        os.makedirs(save_dir, exist_ok=True)
        shape_path = os.path.join(save_dir, f"{filename_base}_shape.glb")
        trimesh_mesh.export(shape_path)
        # Получаем абсолютный путь для формы
        absolute_shape_path = os.path.abspath(shape_path)
        print(f"3D форма сохранена: {absolute_shape_path}")
        
        # Генерация текстуры
        print("Этап 3: Генерация текстуры...")
        textured_mesh = hunyuan3d_texgen_pipe(trimesh_mesh, input_image)
        
        # Сохранение текстурированного меша
        textured_path = os.path.join(save_dir, f"{filename_base}_textured.glb")
        textured_mesh.export(textured_path)
        # Получаем абсолютный путь для текстуры
        absolute_textured_path = os.path.abspath(textured_path)
        print(f"Текстурированный меш сохранен: {absolute_textured_path}")
        
        # Создание объекта Mesh для ComfyUI
        mesh_obj = Mesh(
            v=torch.from_numpy(textured_mesh.vertices).float().to(DEVICE_STR),
            f=torch.from_numpy(textured_mesh.faces).long().to(DEVICE_STR),
            device=DEVICE_STR
        )
        
        print("Полный пайплайн Hunyuan3D завершен успешно!")
        
        return (mesh_obj, absolute_shape_path, absolute_textured_path)

class Multi_Background_Remover:
    """
    Converts 1 to 4 image inputs (front/back/left/right) to a list of processed PIL images.
    Applies RGBA conversion and background removal.
    Suitable for feeding directly into ShapeGen or Paint models.
    """

    CATEGORY = "Comfy3D/Preprocessors"
    RETURN_TYPES = ("LIST",)  # List of PIL images
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_front": ("IMAGE",),
            },
            "optional": {
                "image_back": ("IMAGE",),
                "image_left": ("IMAGE",),
            }
        }

    @torch.no_grad()
    def run(
        self,
        image_front,
        image_back=None,
        image_left=None,
        image_right=None
    ):
        rmbg = BackgroundRemover()

        mv_inputs = {
            k: v for k, v in {
                "front": image_front,
                "back": image_back,
                "left": image_left,
                "right": image_right
            }.items() if v is not None
        }

        images = []
        for key, tensor_img in mv_inputs.items():
            pil_img = torch_imgs_to_pils(tensor_img)[0].convert("RGBA")
            if pil_img.mode == "RGB":
                pil_img = rmbg(pil_img.convert("RGB"))
            images.append(pil_img)

        return (images,)


class Hunyuan3D_Batch_Folder_Pipeline:
    """Батч-обработка папки с картинками через полный пайплайн Hunyuan3D"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_summary",)
    FUNCTION = "process_folder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan3d_pipe": ("HUNYUAN3D_PIPE",),
                "hunyuan3d_texgen_pipe": ("HUNYUAN3D_TEXGEN_PIPE",),
                "input_folder_path": ("STRING", {"forceInput": True}),
                "output_folder_path": ("STRING", {"forceInput": True}),
                "use_background_remover": ("BOOLEAN", {"default": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "postprocess": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "max_files": ("INT", {"default": -1, "min": -1, "max": 1000}),
            }
        }

    def _normalize_filename(self, filename):
        """Нормализует имя файла, убирая лишние нули"""
        import re
        
        # Извлекаем номер из имени файла
        match = re.search(r'(\d+)', filename)
        if match:
            number = int(match.group(1))
            return f"frame_{number}"
        else:
            # Если номер не найден, используем порядковый номер
            return filename

    def _get_sorted_image_files(self, folder_path):
        """Получает отсортированный список PNG файлов"""
        import glob
        
        # Получаем все PNG файлы
        file_paths = glob.glob(os.path.join(folder_path, "*.png"))
        
        # Сортируем по имени файла
        file_paths.sort()
        
        return file_paths

    def _load_image_as_pil(self, image_path):
        """Загружает PNG изображение как PIL"""
        from PIL import Image
        return Image.open(image_path).convert('RGBA')
    
    def _process_with_background_remover(self, pil_image):
        """Обрабатывает изображение через Multi_Background_Remover логику"""
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        # Преобразуем PIL в torch tensor для обработки
        torch_image = pils_to_torch_imgs([pil_image], device=DEVICE_STR)
        
        # Применяем ту же логику что и в Multi_Background_Remover
        rmbg = BackgroundRemover()
        processed_pil = torch_imgs_to_pils(torch_image)[0].convert("RGBA")
        
        if processed_pil.mode == "RGB":
            processed_pil = rmbg(processed_pil.convert("RGB"))
            
        return processed_pil

    def process_folder(self, hunyuan3d_pipe, hunyuan3d_texgen_pipe, input_folder_path, output_folder_path,
                      use_background_remover, num_inference_steps, guidance_scale, octree_resolution,
                      seed, postprocess, max_files=-1):
        
        if not HUNYUAN3D_AVAILABLE:
            raise RuntimeError("Hunyuan3D modules not available")
        
        import glob
        from PIL import Image
        
        print(f"Начинаем батч-обработку папки: {input_folder_path}")
        print(f"Формат файлов: PNG")
        print(f"Выходная папка: {output_folder_path}")
        print(f"Использовать Background Remover: {use_background_remover}")
        
        # Получаем отсортированный список PNG файлов
        image_files = self._get_sorted_image_files(input_folder_path)
        
        if not image_files:
            raise RuntimeError(f"Не найдено PNG изображений в папке {input_folder_path}")
        
        # Ограничиваем количество файлов если задано
        if max_files > 0:
            image_files = image_files[:max_files]
        
        print(f"Найдено PNG файлов для обработки: {len(image_files)}")
        
        # Создаем выходную папку
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Получение устройства из пайплайна
        pipe_device = DEVICE_STR
        if hasattr(hunyuan3d_pipe, 'device'):
            pipe_device = hunyuan3d_pipe.device
        elif hasattr(hunyuan3d_pipe, 'model') and hasattr(hunyuan3d_pipe.model, 'device'):
            pipe_device = hunyuan3d_pipe.model.device
        
        processed_count = 0
        failed_count = 0
        summary_lines = []
        
        for idx, image_path in enumerate(image_files):
            try:
                print(f"\n--- Обработка {idx+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
                
                # Создаем папку для данного фрейма
                frame_name = f"frame_{idx}"
                frame_output_dir = os.path.join(output_folder_path, frame_name)
                os.makedirs(frame_output_dir, exist_ok=True)
                
                # Загружаем изображение
                input_image = self._load_image_as_pil(image_path)
                
                # Применяем Background Remover если включен
                if remove_background:
                    if input_image.mode == 'RGB':
                        input_image = rembg(input_image)
                    print("Фон удален")
                
                # Сохраняем обработанное изображение
                processed_image_path = os.path.join(frame_output_dir, "processed_frame.png")
                input_image.save(processed_image_path)
                
                # Установка seed если указан
                if seed != -1:
                    current_seed = seed + idx  # Разный seed для каждого изображения
                    generator = torch.Generator(device=pipe_device)
                    generator.manual_seed(current_seed)
                else:
                    generator = None
                
                print(f"Генерация 3D формы (seed: {current_seed if seed != -1 else 'random'})...")
                
                # Генерация 3D формы
                mesh_outputs = hunyuan3d_pipe(
                    image=input_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    octree_resolution=octree_resolution,
                    output_type='mesh'
                )
                
                # Преобразование в trimesh
                if isinstance(mesh_outputs, list):
                    trimesh_mesh = export_to_trimesh(mesh_outputs)[0]
                else:
                    trimesh_mesh = export_to_trimesh([mesh_outputs])[0]
                
                # Постобработка если включена
                if postprocess:
                    print("Постобработка меша...")
                    face_reducer = FaceReducer()
                    trimesh_mesh = face_reducer(trimesh_mesh)
                
                # Сохранение формы
                shape_path = os.path.join(frame_output_dir, "shape.glb")
                trimesh_mesh.export(shape_path)
                print(f"3D форма сохранена: {shape_path}")
                
                # Генерация текстуры
                print("Генерация текстуры...")
                textured_mesh = hunyuan3d_texgen_pipe(trimesh_mesh, input_image)
                
                # Сохранение текстурированного меша
                textured_path = os.path.join(frame_output_dir, "textured_mesh.glb")
                textured_mesh.export(textured_path)
                print(f"Текстурированный меш сохранен: {textured_path}")
                
                processed_count += 1
                summary_lines.append(f"✓ {frame_name}: успешно обработан")
                
                print(f"Фрейм {frame_name} обработан успешно!")
                
            except Exception as e:
                failed_count += 1
                error_msg = f"✗ frame_{idx}: ошибка - {str(e)}"
                summary_lines.append(error_msg)
                print(f"Ошибка при обработке {os.path.basename(image_path)}: {e}")
                continue
        
        # Создаем итоговый отчет
        summary = f"""
Батч-обработка завершена!

Статистика:
- Всего файлов: {len(image_files)}
- Успешно обработано: {processed_count}
- Ошибок: {failed_count}

Входная папка: {input_folder}
Выходная папка: {output_folder}

Детали:
""" + "\n".join(summary_lines)
        
        print(summary)
        
        # Сохраняем отчет в файл
        report_path = os.path.join(output_folder, "batch_processing_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return (summary,)