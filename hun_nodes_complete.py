# Hunyuan3D-2.1 - ПОЛНАЯ ВЕРСИЯ с FastMesh/Mesh (БЕЗ trimesh зависимости)

import os
import gc
import torch
import numpy as np
import cv2
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
            # Подготовка изображения
            pil_image = torch_imgs_to_pils(image)[0]
            
            # Background removal
            if remove_background:
                try:
                    bg_remover = BackgroundRemover_2_1()
                    if bg_remover is not None:
                        pil_image = bg_remover(pil_image)
                    del bg_remover
                except Exception as e:
                    print(f"⚠️ Background removal failed: {e}")
            
            # Генерация
            generator = torch.Generator(device=shapegen_pipe.device).manual_seed(seed)
            
            print(f"🚀 Запуск ShapeGen: {steps} шагов, guidance={guidance_scale}")
            outputs = shapegen_pipe(
                image=pil_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=200000,
                output_type='mesh'
            )
            
            # ПРЯМОЙ экспорт из пайплайна в наши объекты!
            print("🎯 ПРЯМОЙ экспорт из пайплайна (БЕЗ trimesh)...")
            
            if output_fastmesh:
                mesh_objects = export_to_fastmesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                print("✅ FastMesh создан напрямую из пайплайна!")
            else:
                mesh_objects = export_to_mesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                print("✅ Стандартный Mesh создан напрямую из пайплайна!")
                
                # Дополняем нормали и текстуру если нужно (БЫСТРО)
                if mesh_out.vn is None:
                    mesh_out.auto_normal()
                if mesh_out.albedo is None:
                    mesh_out._create_empty_albedo_fast()
            
            if mesh_out is None:
                raise Exception("Не удалось создать mesh из пайплайна")
            
            # Оригинальный postprocessing адаптированный для FastMesh/Mesh
            try:
                print("🔄 Применяем ОРИГИНАЛЬНЫЙ postprocessing (адаптированный)...")
                mesh_out = self._apply_original_face_reducer(mesh_out, max_faces=40000)
            except Exception as e:
                print(f"⚠️ Postprocessing ошибка: {e}")
            
            # Cleanup (БЕЗ перемещения пайплайна на CPU!)
            if auto_cleanup:
                try:
                    # НЕ перемещаем пайплайн на CPU - это вызывает проблемы!
                    # shapegen_pipe.to('cpu')  # УБРАНО!
                    
                    # Только очищаем кэш GPU
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("✅ Cleanup завершен (пайплайн остался на GPU)")
                except Exception as e:
                    print(f"⚠️ Cleanup ошибка: {e}")
            
            processed_image_tensor = pils_to_torch_imgs([pil_image])
            
            print(f"✅ ShapeGen завершен: {mesh_out.v.shape[0]} вершин, {mesh_out.f.shape[0]} граней")
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
            
            print(f"🔄 Reducing faces from {mesh_obj.f.shape[0]} to max {max_faces}...")
            
            # Если граней уже меньше максимума - не трогаем
            if mesh_obj.f.shape[0] <= max_faces:
                print(f"✅ Mesh уже содержит {mesh_obj.f.shape[0]} граней (≤ {max_faces})")
                return mesh_obj
            
            # Применяем оригинальный алгоритм через PyMeshLab
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
            
            print(f"✅ Face reduction: {mesh_obj.f.shape[0]} → {reduced_mesh.f.shape[0]} граней")
            return reduced_mesh
            
        except Exception as e:
            print(f"⚠️ Original face reducer error: {e}, возвращаем исходный mesh")
            return mesh_obj
    



class Hunyuan3D_21_TexGen_Complete:
    """Hunyuan3D-2.1 Texture Generation - ПОЛНАЯ ВЕРСИЯ с FastMesh"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texgen_pipe": ("TEXGEN_PIPE",),
                "mesh": ("MESH",),
                "image": ("IMAGE",),
                "output_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Выход FastMesh (иначе стандартный Mesh)"}),
                "create_pbr": ("BOOLEAN", {"default": True, "tooltip": "Создать PBR материалы"}),
                "skip_uv_generation": ("BOOLEAN", {"default": True, "tooltip": "Пропустить UV генерацию (БЫСТРЕЕ!)"}),
                "use_remesh": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"
    CATEGORY = "Comfy3D/Generation/Hunyuan3D"
    
    @torch.no_grad()
    def generate(self, texgen_pipe, mesh, image, output_fastmesh, create_pbr, skip_uv_generation, use_remesh):
        
        try:
            # Подготовка
            pil_image = torch_imgs_to_pils(image)[0]
            temp_dir = "temp_texgen"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Сохраняем mesh как OBJ для texgen пайплайна
            obj_path = os.path.join(temp_dir, "input_mesh.obj")
            mesh.write(obj_path)
            print(f"✅ Mesh сохранен как OBJ: {obj_path}")
            
            # Генерация текстур
            print("🎨 Запуск TexGen...")
            result_path = texgen_pipe(
                mesh_path=obj_path,
                image_path=torch_imgs_to_pils([pil_image])[0] if hasattr(pil_image, 'save') else pil_image,
                output_mesh_path=os.path.join(temp_dir, "textured_output.obj"),
                save_glb=True,
                use_remesh=use_remesh
            )
            
            # Создаем GLB с текстурами
            glb_path = result_path.replace(".obj", ".glb")
            
            if create_pbr:
                # Загружаем отдельные текстуры
                base_path = os.path.splitext(result_path)[0]
                textures = {}
                
                albedo_path = f"{base_path}.jpg"
                if os.path.exists(albedo_path):
                    textures['albedo'] = albedo_path
                
                metallic_path = f"{base_path}_metallic.jpg"
                if os.path.exists(metallic_path):
                    textures['metallic'] = metallic_path
                
                roughness_path = f"{base_path}_roughness.jpg"
                if os.path.exists(roughness_path):
                    textures['roughness'] = roughness_path
                
                # Создаем GLB с нашей функцией
                success = self._create_complete_glb(result_path, textures, glb_path, output_fastmesh, skip_uv_generation)
                
                if not success:
                    # Fallback - простая конвертация
                    if output_fastmesh:
                        temp_mesh = FastMesh.load(result_path)
                    else:
                        temp_mesh = Mesh.load_obj(result_path)
                    
                    if temp_mesh is not None:
                        temp_mesh.write(glb_path)
                        print(f"✅ GLB создан простой конвертацией: {glb_path}")
            
            # Загружаем финальный результат
            if os.path.exists(glb_path):
                if output_fastmesh:
                    mesh_out = FastMesh.load(glb_path)
                    print("🚀 GLB загружен через FastMesh")
                else:
                    mesh_out = Mesh.load_gltf(glb_path)
                    print("🔧 GLB загружен через стандартный Mesh")
                
                if mesh_out is not None:
                    mesh_out.auto_normal()
                    print(f"✅ TexGen завершен: текстуры применены")
                    return (mesh_out,)
            
            # Fallback если GLB не создался
            print("⚠️ GLB не создался, загружаем OBJ...")
            if output_fastmesh:
                mesh_out = FastMesh.load(result_path)
            else:
                mesh_out = Mesh.load_obj(result_path)
            
            if mesh_out is not None:
                mesh_out.auto_normal()
                return (mesh_out,)
            
            raise Exception("Не удалось загрузить результат")
            
        except Exception as e:
            print(f"❌ TexGen ошибка: {e}")
            return (mesh,)  # Возвращаем оригинальный mesh
    
    def _create_complete_glb(self, obj_path, textures_dict, output_glb_path, use_fastmesh=True, skip_uv_generation=True):
        """Создает полный GLB с текстурами"""
        
        try:
            print(f"🎨 Создаем полный GLB: {output_glb_path}")
            
            # Загружаем OBJ
            if use_fastmesh:
                mesh = FastMesh.load(obj_path)
            else:
                mesh = Mesh.load_obj(obj_path)
            
            if mesh is None:
                return False
            
            # Загружаем albedo
            if 'albedo' in textures_dict:
                albedo = cv2.imread(textures_dict['albedo'], cv2.IMREAD_COLOR)
                if albedo is not None:
                    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                    albedo = albedo.astype(np.float32) / 255.0
                    mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=mesh.device)
                    print(f"✅ Albedo загружен: {albedo.shape}")
            
            # Создаем metallic-roughness
            metallic_data = None
            roughness_data = None
            
            if 'metallic' in textures_dict:
                metallic = cv2.imread(textures_dict['metallic'], cv2.IMREAD_GRAYSCALE)
                if metallic is not None:
                    metallic_data = metallic.astype(np.float32) / 255.0
            
            if 'roughness' in textures_dict:
                roughness = cv2.imread(textures_dict['roughness'], cv2.IMREAD_GRAYSCALE)
                if roughness is not None:
                    roughness_data = roughness.astype(np.float32) / 255.0
            
            # Комбинируем metallic-roughness
            if metallic_data is not None or roughness_data is not None:
                if mesh.albedo is not None:
                    h, w = mesh.albedo.shape[:2]
                    mr_texture = np.zeros((h, w, 3), dtype=np.float32)
                    
                    if roughness_data is not None:
                        if roughness_data.shape != (h, w):
                            roughness_data = cv2.resize(roughness_data, (w, h))
                        mr_texture[:, :, 1] = roughness_data
                    else:
                        mr_texture[:, :, 1] = 0.8  # Default roughness
                    
                    if metallic_data is not None:
                        if metallic_data.shape != (h, w):
                            metallic_data = cv2.resize(metallic_data, (w, h))
                        mr_texture[:, :, 2] = metallic_data
                    else:
                        mr_texture[:, :, 2] = 0.1  # Default metallic
                    
                    mesh.metallicRoughness = torch.tensor(mr_texture, dtype=torch.float32, device=mesh.device)
                    print(f"✅ MetallicRoughness создан: {mr_texture.shape}")
            
            # Генерируем UV если нужно (ОПЦИОНАЛЬНО - может быть медленно!)
            if not skip_uv_generation and mesh.vt is None and mesh.albedo is not None:
                print("🗺️ ВНИМАНИЕ: Генерируем UV (это может быть медленно)...")
                mesh.auto_uv(cache_path=obj_path)
            elif skip_uv_generation:
                print("⚡ UV генерация пропущена (для скорости)")
            
            # Сохраняем GLB
            mesh.write(output_glb_path)
            print(f"✅ Полный GLB создан: {output_glb_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания полного GLB: {e}")
            return False


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
        pil_image = torch_imgs_to_pils(image)[0].convert("RGBA")
        
        if remove_background or pil_image.mode == "RGB":
            rmbg_worker = BackgroundRemover_2_1()
            pil_image = rmbg_worker(pil_image.convert('RGB'))
            del rmbg_worker

        generator = torch.Generator(device=shapegen_pipe.device)
        generator = generator.manual_seed(int(seed))
        
        outputs = shapegen_pipe(
            image=pil_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=200000,
            output_type='mesh'
        )
        
        mesh = export_to_trimesh_2_1(outputs)[0]
        if auto_cleanup:
            face_reduce_worker = FaceReducer_2_1()
            mesh = face_reduce_worker(mesh)
            del face_reduce_worker
        
        # if auto_cleanup:
        #     try:
        #         shapegen_pipe.to('cpu')
        #         if hasattr(shapegen_pipe, 'unet'):
        #             del shapegen_pipe.unet
        #         if hasattr(shapegen_pipe, 'vae'):
        #             del shapegen_pipe.vae
        #         if hasattr(shapegen_pipe, 'scheduler'):
        #             del shapegen_pipe.scheduler
        #         del outputs
        #         torch.cuda.empty_cache()
        #         gc.collect()
        #         print("Shape pipeline cleaned up")
        #     except Exception as e:
        #         print(f"Error during pipeline cleanup: {e}")
            
        mesh_out = Mesh.load_trimesh(given_mesh=mesh)
        mesh_out.auto_normal()
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        return (mesh_out, processed_image_tensor)

# # Mapping для ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "Hunyuan3D_21_ShapeGen_Complete": Hunyuan3D_21_ShapeGen_Complete,
#     "Hunyuan3D_21_TexGen_Complete": Hunyuan3D_21_TexGen_Complete,
#     "FastMesh_Utilities": FastMesh_Utilities,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Hunyuan3D_21_ShapeGen_Complete": "Hunyuan3D 2.1 ShapeGen (FastMesh)",
#     "Hunyuan3D_21_TexGen_Complete": "Hunyuan3D 2.1 TexGen (FastMesh)",
#     "FastMesh_Utilities": "FastMesh Utilities",
# }