# Hunyuan3D-2.1 - ПОЛНАЯ ВЕРСИЯ с FastMesh/Mesh (БЕЗ trimesh зависимости)

import os
import gc
import torch
import numpy as np
import cv2
import time
from typing import Union
from mesh_processor.export_utils import export_to_fastmesh, export_to_mesh
from mesh_processor.mesh import FastMesh, Mesh
from fastpostprocessors import FastMeshCleaner, fast_reduce_faces

# Импорты с fallback
try:
    from shared_utils.image_utils import torch_imgs_to_pils, pils_to_torch_imgs
    from Gen_3D_Modules.Hunyuan3D_2_1 import BackgroundRemover_2_1
except ImportError as e:
    print(f"⚠️ Импорты недоступны: {e}")
    def torch_imgs_to_pils(imgs): return []
    def pils_to_torch_imgs(pils): return torch.tensor([])
    def BackgroundRemover_2_1(): return None


class Hunyuan3D_21_ShapeGen_Complete:
    """Hunyuan3D-2.1 Shape Generation - ПОЛНАЯ ВЕРСИЯ с FastMesh"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shapegen_pipe": ("DIFFUSERS_PIPE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.1, "max": 30.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64}),
                "output_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Выход FastMesh (иначе стандартный Mesh)"}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "auto_cleanup": ("BOOLEAN", {"default": True}),
                "keep_pipeline_on_gpu": ("BOOLEAN", {"default": True, "tooltip": "Оставить пайплайн на GPU для быстрых повторных запусков"}),
            },
        }
    
    RETURN_TYPES = ("MESH", "IMAGE")
    RETURN_NAMES = ("mesh", "image")
    FUNCTION = "generate"
    CATEGORY = "Comfy3D/Generation/Hunyuan3D"
    
    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, output_fastmesh, remove_background, auto_cleanup, keep_pipeline_on_gpu):
        
        try:
            total_start = time.time()
            print(f"🏁 Начинаем полный цикл ShapeGen...")
            
            # Подготовка изображения
            prep_start = time.time()
            pil_image = torch_imgs_to_pils(image)[0]
            prep_time = time.time() - prep_start
            print(f"⏱️ Подготовка изображения: {prep_time:.3f}с")
            
            # Background removal
            bg_start = time.time()
            if remove_background:
                try:
                    bg_remover = BackgroundRemover_2_1()
                    if bg_remover is not None:
                        pil_image = bg_remover(pil_image)
                    del bg_remover
                except Exception as e:
                    print(f"⚠️ Background removal failed: {e}")
            bg_time = time.time() - bg_start
            print(f"⏱️ Background removal: {bg_time:.3f}с")
            
            # Генератор
            gen_setup_start = time.time()
            generator = torch.Generator(device=shapegen_pipe.device).manual_seed(seed)
            gen_setup_time = time.time() - gen_setup_start
            print(f"⏱️ Настройка генератора: {gen_setup_time:.3f}с")
            
            print(f"🚀 Запуск ShapeGen: {steps} шагов, guidance={guidance_scale}")
            generation_start = time.time()
            outputs = shapegen_pipe(
                image=pil_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=200000,
                output_type='mesh'
            )
            generation_time = time.time() - generation_start
            print(f"⏱️ ShapeGen время генерации: {generation_time:.2f}с")
            
            # ПРЯМОЙ экспорт из пайплайна в наши объекты!
            print("🎯 ПРЯМОЙ экспорт из пайплайна (БЕЗ trimesh)...")
            export_start = time.time()
            
            if output_fastmesh:
                export_type_start = time.time()
                mesh_objects = export_to_fastmesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                export_type_time = time.time() - export_type_start
                print(f"✅ FastMesh создан напрямую из пайплайна! ({export_type_time:.2f}с)")
            else:
                export_type_start = time.time()
                mesh_objects = export_to_mesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                export_type_time = time.time() - export_type_start
                print(f"✅ Стандартный Mesh создан напрямую из пайплайна! ({export_type_time:.2f}с)")
                
                # Дополняем нормали и текстуру если нужно (БЫСТРО)
                post_start = time.time()
                if mesh_out.vn is None:
                    mesh_out.auto_normal()
                if mesh_out.albedo is None:
                    mesh_out._create_empty_albedo_fast()
                post_time = time.time() - post_start
                print(f"⚡ Постобработка меша: {post_time:.3f}с")
            
            export_time = time.time() - export_start
            print(f"⏱️ Общее время экспорта: {export_time:.2f}с")
            
            if mesh_out is None:
                raise Exception("Не удалось создать mesh из пайплайна")
            
            # Оригинальный postprocessing адаптированный для FastMesh/Mesh
            try:
                print("🔄 Применяем ОРИГИНАЛЬНЫЙ postprocessing (адаптированный)...")
                reduction_start = time.time()
                mesh_out = self._apply_original_face_reducer(mesh_out, max_faces=40000)
                reduction_time = time.time() - reduction_start
                print(f"⏱️ Face reduction время: {reduction_time:.2f}с")
            except Exception as e:
                print(f"⚠️ Postprocessing ошибка: {e}")
            
            # Cleanup (БЕЗ перемещения пайплайна на CPU!)
            if auto_cleanup:
                try:
                    cleanup_start = time.time()
                    # НЕ перемещаем пайплайн на CPU - это вызывает проблемы!
                    # shapegen_pipe.to('cpu')  # УБРАНО!
                    
                    # Только очищаем кэш GPU
                    torch.cuda.empty_cache()
                    gc.collect()
                    cleanup_time = time.time() - cleanup_start
                    print(f"✅ Cleanup завершен (пайплайн остался на GPU) - {cleanup_time:.3f}с")
                except Exception as e:
                    print(f"⚠️ Cleanup ошибка: {e}")
            
            processed_image_tensor = pils_to_torch_imgs([pil_image])
            
            total_time = time.time() - total_start
            print(f"🏆 ИТОГОВАЯ СВОДКА (НОВЫЙ FastMesh):")
            print(f"  📊 Результат: {mesh_out.v.shape[0]} вершин, {mesh_out.f.shape[0]} граней")
            print(f"  ⏱️ ОБЩЕЕ ВРЕМЯ: {total_time:.2f}с")
            print(f"  🔹 Подготовка изображения: {prep_time:.3f}с ({prep_time/total_time*100:.1f}%)")
            print(f"  🔹 Background removal: {bg_time:.3f}с ({bg_time/total_time*100:.1f}%)")
            print(f"  🔹 Настройка генератора: {gen_setup_time:.3f}с ({gen_setup_time/total_time*100:.1f}%)")
            print(f"  🔹 Генерация: {generation_time:.2f}с ({generation_time/total_time*100:.1f}%)")
            print(f"  🔹 Экспорт: {export_time:.2f}с ({export_time/total_time*100:.1f}%)")
            if 'reduction_time' in locals():
                print(f"  🔹 Face reduction: {reduction_time:.2f}с ({reduction_time/total_time*100:.1f}%)")
            if 'cleanup_time' in locals():
                print(f"  🔹 Cleanup: {cleanup_time:.3f}с ({cleanup_time/total_time*100:.1f}%)")
            return (mesh_out, processed_image_tensor)
            
        except Exception as e:
            print(f"❌ ShapeGen ошибка: {e}")
            return (None, pils_to_torch_imgs([pil_image]) if 'pil_image' in locals() else torch.zeros((1, 3, 512, 512)))
    
    def _apply_original_face_reducer(self, mesh_obj: Union[Mesh, FastMesh], max_faces: int = 40000) -> Union[Mesh, FastMesh]:
        """
        Применяет оригинальный FaceReducer алгоритм, но БЕЗ trimesh зависимости
        Использует PyMeshLab через наш wrapper
        """
        try:
            # Используем наш PyMeshLab wrapper вместо оригинального trimesh подхода
            from pymeshlab_wrapper import pymeshlab_reduce_faces
            
            faces_before = mesh_obj.f.shape[0]
            print(f"🔄 Reducing faces from {faces_before} to max {max_faces}...")
            
            # Если граней уже меньше максимума - не трогаем
            if faces_before <= max_faces:
                print(f"✅ Mesh уже содержит {faces_before} граней (≤ {max_faces}) - пропускаем")
                return mesh_obj
            
            # Применяем оригинальный алгоритм через PyMeshLab
            algo_start = time.time()
            reduced_mesh = pymeshlab_reduce_faces(
                mesh_obj, 
                max_faces=max_faces,
                quality_threshold=1.0,
                preserve_boundary=True,
                boundary_weight=3,
                preserve_normal=True,
                preserve_topology=True,
                autoclean=True
            )
            algo_time = time.time() - algo_start
            
            faces_after = reduced_mesh.f.shape[0]
            reduction_ratio = ((faces_before - faces_after) / faces_before) * 100
            print(f"✅ Face reduction: {faces_before} → {faces_after} граней (-{reduction_ratio:.1f}%) за {algo_time:.2f}с")
            return reduced_mesh
            
        except Exception as e:
            print(f"⚠️ Original face reducer error: {e}, возвращаем исходный mesh")
            return mesh_obj
    

class FastMesh_Utilities:
    """Утилиты для работы с FastMesh"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "operation": (["info", "optimize", "clean", "auto_uv", "to_fastmesh", "to_standard_mesh"], {"default": "info"}),
            },
        }
    
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "process"
    CATEGORY = "Comfy3D/Utilities"
    
    def process(self, mesh, operation):
        
        info = ""
        mesh_out = mesh
        
        try:
            if operation == "info":
                info = self._get_mesh_info(mesh)
                
            elif operation == "optimize":
                mesh_out = self._optimize_mesh(mesh)
                info = "Mesh оптимизирован"
                
            elif operation == "clean":
                mesh_out = self._clean_mesh(mesh)
                info = "Mesh очищен"
                
            elif operation == "auto_uv":
                mesh_out = self._generate_uv(mesh)
                info = "UV координаты сгенерированы"
                
            elif operation == "to_fastmesh":
                if isinstance(mesh, FastMesh):
                    info = "Mesh уже FastMesh"
                else:
                    mesh_out = self._convert_to_fastmesh(mesh)
                    info = "Конвертирован в FastMesh"
                    
            elif operation == "to_standard_mesh":
                if isinstance(mesh, Mesh) and not isinstance(mesh, FastMesh):
                    info = "Mesh уже стандартный"
                else:
                    mesh_out = self._convert_to_standard_mesh(mesh)
                    info = "Конвертирован в стандартный Mesh"
            
        except Exception as e:
            info = f"Ошибка: {e}"
            
        return (mesh_out, info)
    
    def _get_mesh_info(self, mesh):
        """Получает информацию о mesh"""
        
        info_lines = []
        
        # Тип mesh
        if isinstance(mesh, FastMesh):
            info_lines.append("🚀 Тип: FastMesh")
        elif isinstance(mesh, Mesh):
            info_lines.append("🔧 Тип: Стандартный Mesh")
        else:
            info_lines.append(f"❓ Тип: {type(mesh).__name__}")
        
        # Геометрия
        if hasattr(mesh, 'v') and mesh.v is not None:
            info_lines.append(f"📐 Вершины: {mesh.v.shape[0]}")
        
        if hasattr(mesh, 'f') and mesh.f is not None:
            info_lines.append(f"🔺 Грани: {mesh.f.shape[0]}")
        
        # Атрибуты
        attrs = []
        if hasattr(mesh, 'vn') and mesh.vn is not None:
            attrs.append(f"Нормали: {mesh.vn.shape[0]}")
        
        if hasattr(mesh, 'vt') and mesh.vt is not None:
            attrs.append(f"UV: {mesh.vt.shape[0]}")
        
        if hasattr(mesh, 'vc') and mesh.vc is not None:
            attrs.append(f"Цвета: {mesh.vc.shape[0]}")
        
        if attrs:
            info_lines.append(f"📊 Атрибуты: {', '.join(attrs)}")
        
        # Текстуры
        textures = []
        if hasattr(mesh, 'albedo') and mesh.albedo is not None:
            textures.append(f"Albedo: {mesh.albedo.shape}")
        
        if hasattr(mesh, 'metallicRoughness') and mesh.metallicRoughness is not None:
            textures.append(f"MetallicRoughness: {mesh.metallicRoughness.shape}")
        
        if textures:
            info_lines.append(f"🎨 Текстуры: {', '.join(textures)}")
        
        # Device
        if hasattr(mesh, 'device'):
            info_lines.append(f"💾 Device: {mesh.device}")
        
        return "\n".join(info_lines)
    
    def _optimize_mesh(self, mesh):
        """Оптимизирует mesh"""
        # Удаляем дубликаты вершин, пересчитываем нормали, UV
        if hasattr(mesh, 'auto_normal'):
            mesh.auto_normal()
        return mesh
    
    def _clean_mesh(self, mesh):
        """Очищает mesh"""
        if hasattr(mesh, '_clean_mesh'):
            mesh._clean_mesh()
        return mesh
    
    def _generate_uv(self, mesh):
        """Генерирует UV координаты"""
        if hasattr(mesh, 'auto_uv'):
            mesh.auto_uv()
        return mesh
    
    def _convert_to_fastmesh(self, mesh):
        """Конвертирует в FastMesh"""
        if isinstance(mesh, FastMesh):
            return mesh
        
        return FastMesh(
            v=mesh.v,
            f=mesh.f,
            vn=mesh.vn,
            fn=mesh.fn,
            vt=mesh.vt,
            ft=mesh.ft,
            vc=mesh.vc,
            albedo=mesh.albedo,
            metallicRoughness=mesh.metallicRoughness,
            device=mesh.device
        )
    
    def _convert_to_standard_mesh(self, mesh):
        """Конвертирует в стандартный Mesh"""
        if isinstance(mesh, Mesh) and not isinstance(mesh, FastMesh):
            return mesh
        
        return Mesh(
            v=mesh.v,
            f=mesh.f,
            vn=mesh.vn,
            fn=mesh.fn,
            vt=mesh.vt,
            ft=mesh.ft,
            vc=mesh.vc,
            albedo=mesh.albedo,
            metallicRoughness=mesh.metallicRoughness,
            device=mesh.device
        )

class Hunyuan3D_21_ShapeGen:
    """Hunyuan3D-2.1 Shape Generation with automatic pipeline cleanup"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("MESH", "IMAGE")
    RETURN_NAMES = ("mesh", "processed_image")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shapegen_pipe": ("DIFFUSERS_PIPE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 512}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "auto_cleanup": ("BOOLEAN", {"default": True}),
            }
        }

    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, remove_background, auto_cleanup):
        total_start = time.time()
        print(f"🏁 Начинаем ОРИГИНАЛЬНЫЙ цикл ShapeGen (с trimesh)...")
        
        # Подготовка изображения
        prep_start = time.time()
        pil_image = torch_imgs_to_pils(image)[0].convert("RGBA")
        prep_time = time.time() - prep_start
        print(f"⏱️ Подготовка изображения: {prep_time:.3f}с")
        
        # Background removal
        bg_start = time.time()
        
        if remove_background or pil_image.mode == "RGB":
            rmbg_worker = BackgroundRemover_2_1()
            pil_image = rmbg_worker(pil_image.convert('RGB'))
            del rmbg_worker
        bg_time = time.time() - bg_start
        print(f"⏱️ Background removal: {bg_time:.3f}с")

        # Генератор
        gen_setup_start = time.time()
        generator = torch.Generator(device=shapegen_pipe.device)
        generator = generator.manual_seed(int(seed))
        gen_setup_time = time.time() - gen_setup_start
        print(f"⏱️ Настройка генератора: {gen_setup_time:.3f}с")
        
        print(f"🚀 Запуск ОРИГИНАЛЬНОГО ShapeGen: {steps} шагов, guidance={guidance_scale}")
        generation_start = time.time()
        outputs = shapegen_pipe(
            image=pil_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=200000,
            output_type='mesh'
        )
        generation_time = time.time() - generation_start
        print(f"⏱️ ShapeGen время генерации: {generation_time:.2f}с")
        
        # Экспорт в trimesh
        export_start = time.time()
        mesh = export_to_trimesh_2_1(outputs)[0]
        export_time = time.time() - export_start
        print(f"⏱️ Экспорт в trimesh: {export_time:.2f}с")
        
        # Face reduction (если включен)
        if auto_cleanup:
            faces_before = len(mesh.faces)
            reduction_start = time.time()
            face_reduce_worker = FaceReducer_2_1()
            mesh = face_reduce_worker(mesh)
            del face_reduce_worker
            reduction_time = time.time() - reduction_start
            faces_after = len(mesh.faces)
            reduction_ratio = ((faces_before - faces_after) / faces_before) * 100 if faces_before > 0 else 0
            print(f"⏱️ ОРИГИНАЛЬНЫЙ Face reduction: {faces_before} → {faces_after} граней (-{reduction_ratio:.1f}%) за {reduction_time:.2f}с")
        else:
            reduction_time = 0
        
        # Конвертация trimesh → Mesh  
        convert_start = time.time()
        mesh_out = Mesh.load_trimesh(given_mesh=mesh)
        mesh_out.auto_normal()
        convert_time = time.time() - convert_start
        print(f"⏱️ Конвертация trimesh → Mesh: {convert_time:.3f}с")
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        # Итоговая сводка
        total_time = time.time() - total_start
        print(f"🏆 ИТОГОВАЯ СВОДКА (ОРИГИНАЛЬНЫЙ trimesh):")
        print(f"  📊 Результат: {mesh_out.v.shape[0]} вершин, {mesh_out.f.shape[0]} граней")
        print(f"  ⏱️ ОБЩЕЕ ВРЕМЯ: {total_time:.2f}с")
        print(f"  🔹 Подготовка изображения: {prep_time:.3f}с ({prep_time/total_time*100:.1f}%)")
        print(f"  🔹 Background removal: {bg_time:.3f}с ({bg_time/total_time*100:.1f}%)")
        print(f"  🔹 Настройка генератора: {gen_setup_time:.3f}с ({gen_setup_time/total_time*100:.1f}%)")
        print(f"  🔹 Генерация: {generation_time:.2f}с ({generation_time/total_time*100:.1f}%)")
        print(f"  🔹 Экспорт в trimesh: {export_time:.2f}с ({export_time/total_time*100:.1f}%)")
        if auto_cleanup:
            print(f"  🔹 Face reduction: {reduction_time:.2f}с ({reduction_time/total_time*100:.1f}%)")
        print(f"  🔹 Конвертация → Mesh: {convert_time:.3f}с ({convert_time/total_time*100:.1f}%)")
        
        return (mesh_out, processed_image_tensor)
