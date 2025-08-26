# Hunyuan3D-2.1 - ОПТИМИЗИРОВАННАЯ версия БЕЗ медленных операций

import os
import gc
import torch
import numpy as np
import cv2
from mesh_processor.export_utils import export_to_fastmesh, export_to_mesh
from mesh_processor.mesh import FastMesh, Mesh

# Импорты с fallback
try:
    from shared_utils.image_utils import torch_imgs_to_pils, pils_to_torch_imgs
    from Gen_3D_Modules.Hunyuan3D_2_1 import BackgroundRemover_2_1
except ImportError as e:
    print(f"⚠️ Импорты недоступны: {e}")
    def torch_imgs_to_pils(imgs): return []
    def pils_to_torch_imgs(pils): return torch.tensor([])
    def BackgroundRemover_2_1(): return None


class Hunyuan3D_21_ShapeGen_Optimized:
    """Hunyuan3D-2.1 Shape Generation - ОПТИМИЗИРОВАННАЯ без медленных операций"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shapegen_pipe": ("SHAPEGEN_PIPE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.1, "max": 30.0, "step": 0.1}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64}),
                "output_fastmesh": ("BOOLEAN", {"default": True}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "skip_uv_generation": ("BOOLEAN", {"default": True, "tooltip": "Пропустить генерацию UV (быстрее)"}),
                "skip_face_reduction": ("BOOLEAN", {"default": True, "tooltip": "Пропустить face reduction (быстрее)"}),
                "auto_cleanup": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MESH", "IMAGE")
    RETURN_NAMES = ("mesh", "image")
    FUNCTION = "generate"
    CATEGORY = "Comfy3D/Generation/Hunyuan3D"
    
    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, 
                 output_fastmesh, remove_background, skip_uv_generation, skip_face_reduction, auto_cleanup):
        
        try:
            print("🚀 Запуск ОПТИМИЗИРОВАННОГО ShapeGen...")
            
            # Подготовка изображения
            pil_image = torch_imgs_to_pils(image)[0]
            
            # Background removal (быстрый)
            if remove_background:
                try:
                    bg_remover = BackgroundRemover_2_1()
                    if bg_remover is not None:
                        pil_image = bg_remover(pil_image)
                    del bg_remover
                except Exception as e:
                    print(f"⚠️ Background removal failed: {e}")
            
            # Генерация (основной процесс)
            generator = torch.Generator(device=shapegen_pipe.device).manual_seed(seed)
            
            print(f"🎯 Генерация меша: {steps} шагов, guidance={guidance_scale}")
            outputs = shapegen_pipe(
                image=pil_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=200000,
                output_type='mesh'
            )
            
            # ПРЯМОЙ экспорт БЕЗ медленных операций
            print("⚡ БЫСТРЫЙ экспорт из пайплайна...")
            
            if output_fastmesh:
                mesh_objects = export_to_fastmesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                print("✅ FastMesh создан напрямую!")
            else:
                mesh_objects = export_to_mesh(outputs)
                mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
                print("✅ Стандартный Mesh создан напрямую!")
            
            if mesh_out is None:
                raise Exception("Не удалось создать mesh из пайплайна")
            
            print(f"📊 Исходный меш: {mesh_out.v.shape[0]} вершин, {mesh_out.f.shape[0]} граней")
            
            # Face reduction (ТОЛЬКО если не пропускаем)
            if not skip_face_reduction:
                try:
                    print("🔧 Быстрое face reduction...")
                    mesh_out = self._fast_face_reduction(mesh_out, reduction_factor=0.8)
                except Exception as e:
                    print(f"⚠️ Face reduction ошибка: {e}")
            else:
                print("⚡ Face reduction пропущен (для скорости)")
            
            # UV генерация (ТОЛЬКО если не пропускаем)
            if not skip_uv_generation and mesh_out.vt is None:
                print("🗺️ ВНИМАНИЕ: Генерация UV может быть медленной...")
                try:
                    mesh_out.auto_uv()
                except Exception as e:
                    print(f"⚠️ UV generation ошибка: {e}")
            else:
                print("⚡ UV генерация пропущена (для скорости)")
            
            # Нормали (быстро)
            if mesh_out.vn is None:
                mesh_out.auto_normal()
                print("✅ Нормали вычислены")
            
            # Cleanup
            if auto_cleanup:
                try:
                    shapegen_pipe.to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("✅ Cleanup завершен")
                except Exception as e:
                    print(f"⚠️ Cleanup ошибка: {e}")
            
            processed_image_tensor = pils_to_torch_imgs([pil_image])
            
            print(f"🎉 ShapeGen завершен за минимальное время!")
            print(f"📊 Финальный меш: {mesh_out.v.shape[0]} вершин, {mesh_out.f.shape[0]} граней")
            
            return (mesh_out, processed_image_tensor)
            
        except Exception as e:
            print(f"❌ ShapeGen ошибка: {e}")
            return (None, pils_to_torch_imgs([pil_image]) if 'pil_image' in locals() else torch.zeros((1, 3, 512, 512)))
    
    def _fast_face_reduction(self, mesh_obj, reduction_factor=0.8):
        """Быстрое face reduction по площади треугольников"""
        try:
            vertices = mesh_obj.v
            faces = mesh_obj.f
            
            if len(faces) < 1000:  # Если меш маленький, не трогаем
                print("Меш уже маленький, face reduction пропущен")
                return mesh_obj
            
            # Быстрый расчет площадей
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]] 
            v2 = vertices[faces[:, 2]]
            
            # Векторное произведение для площади
            cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
            areas = torch.norm(cross_product, dim=1)
            
            # Простая сортировка и отбор лучших
            _, sorted_indices = torch.sort(areas, descending=True)
            keep_count = int(len(sorted_indices) * reduction_factor)
            keep_faces = sorted_indices[:keep_count]
            
            new_faces = faces[keep_faces]
            mesh_obj.f = new_faces
            
            if mesh_obj.fn is not None:
                mesh_obj.fn = new_faces
            
            print(f"⚡ Быстрое face reduction: {len(faces)} → {len(new_faces)} граней")
            
        except Exception as e:
            print(f"Face reduction ошибка: {e}")
        
        return mesh_obj


class Hunyuan3D_21_TexGen_Optimized:
    """Hunyuan3D-2.1 Texture Generation - ОПТИМИЗИРОВАННАЯ версия"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texgen_pipe": ("TEXGEN_PIPE",),
                "mesh": ("MESH",),
                "image": ("IMAGE",),
                "output_fastmesh": ("BOOLEAN", {"default": True}),
                "create_pbr": ("BOOLEAN", {"default": True}),
                "skip_uv_regeneration": ("BOOLEAN", {"default": True, "tooltip": "Пропустить перегенерацию UV (быстрее)"}),
                "use_remesh": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"
    CATEGORY = "Comfy3D/Generation/Hunyuan3D"
    
    @torch.no_grad()
    def generate(self, texgen_pipe, mesh, image, output_fastmesh, create_pbr, skip_uv_regeneration, use_remesh):
        
        try:
            print("🎨 Запуск ОПТИМИЗИРОВАННОГО TexGen...")
            
            # Подготовка
            pil_image = torch_imgs_to_pils(image)[0]
            temp_dir = "temp_texgen"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Сохраняем mesh как OBJ для texgen пайплайна
            obj_path = os.path.join(temp_dir, "input_mesh.obj")
            mesh.write(obj_path)
            print(f"✅ Mesh сохранен: {obj_path}")
            
            # Генерация текстур (основной процесс)
            print("🎨 Запуск генерации текстур...")
            result_path = texgen_pipe(
                mesh_path=obj_path,
                image_path=pil_image,
                output_mesh_path=os.path.join(temp_dir, "textured_output.obj"),
                save_glb=True,
                use_remesh=use_remesh
            )
            
            # Быстрое создание GLB
            glb_path = result_path.replace(".obj", ".glb")
            
            if create_pbr:
                # Загружаем текстуры БЫСТРО
                success = self._create_fast_glb(result_path, glb_path, output_fastmesh, skip_uv_regeneration)
                
                if not success:
                    print("⚠️ Быстрое создание GLB failed, простая конвертация...")
                    # Простая fallback конвертация
                    if output_fastmesh:
                        temp_mesh = FastMesh.load(result_path)
                    else:
                        temp_mesh = Mesh.load_obj(result_path)
                    
                    if temp_mesh is not None:
                        temp_mesh.write(glb_path)
            
            # Загружаем результат
            if os.path.exists(glb_path):
                print(f"📁 Загружаем GLB: {glb_path}")
                
                if output_fastmesh:
                    mesh_out = FastMesh.load(glb_path)
                    print("🚀 GLB загружен через FastMesh")
                else:
                    mesh_out = Mesh.load_gltf(glb_path)
                    print("🔧 GLB загружен через стандартный Mesh")
                
                if mesh_out is not None:
                    # Только быстрые операции
                    mesh_out.auto_normal()
                    print(f"🎉 TexGen завершен: {mesh_out.v.shape[0]} вершин с текстурами!")
                    return (mesh_out,)
            
            # Fallback к OBJ
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
    
    def _create_fast_glb(self, obj_path, output_glb_path, use_fastmesh, skip_uv_regeneration):
        """Быстрое создание GLB с минимальными операциями"""
        
        try:
            print(f"⚡ Быстрое создание GLB: {output_glb_path}")
            
            # Загружаем OBJ быстро
            if use_fastmesh:
                mesh = FastMesh.load(obj_path)
            else:
                mesh = Mesh.load_obj(obj_path)
            
            if mesh is None:
                return False
            
            # Загружаем ТОЛЬКО albedo (самое важное)
            base_path = os.path.splitext(obj_path)[0]
            albedo_path = f"{base_path}.jpg"
            
            if os.path.exists(albedo_path):
                albedo = cv2.imread(albedo_path, cv2.IMREAD_COLOR)
                if albedo is not None:
                    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                    albedo = albedo.astype(np.float32) / 255.0
                    mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=mesh.device)
                    print(f"✅ Albedo загружен: {albedo.shape}")
                    
                    # Создаем базовый metallic-roughness БЕЗ загрузки файлов
                    h, w = albedo.shape[:2]
                    mr_texture = np.zeros((h, w, 3), dtype=np.float32)
                    mr_texture[:, :, 1] = 0.8  # Roughness
                    mr_texture[:, :, 2] = 0.1  # Metallic
                    mesh.metallicRoughness = torch.tensor(mr_texture, dtype=torch.float32, device=mesh.device)
                    print("✅ Базовый MetallicRoughness создан")
            
            # UV только если нужно И не пропускаем
            if mesh.vt is None and mesh.albedo is not None and not skip_uv_regeneration:
                print("🗺️ ВНИМАНИЕ: Генерация UV...")
                mesh.auto_uv(cache_path=obj_path)
            elif skip_uv_regeneration:
                print("⚡ UV регенерация пропущена (для скорости)")
            
            # Сохраняем GLB
            mesh.write(output_glb_path)
            print(f"⚡ Быстрый GLB создан: {output_glb_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка быстрого GLB: {e}")
            return False


class FastMesh_Simple_Loader:
    """Простой загрузчик mesh БЕЗ медленных операций"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "use_fastmesh": ("BOOLEAN", {"default": True}),
                "skip_uv_generation": ("BOOLEAN", {"default": True, "tooltip": "Пропустить UV (быстрее)"}),
                "skip_texture_loading": ("BOOLEAN", {"default": False, "tooltip": "Пропустить текстуры (быстрее)"}),
            },
        }
    
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_fast"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_fast(self, mesh_file_path, use_fastmesh, skip_uv_generation, skip_texture_loading):
        
        info = ""
        mesh_out = None
        
        try:
            if not os.path.exists(mesh_file_path):
                info = f"❌ Файл не найден: {mesh_file_path}"
                return (None, info)
            
            print(f"⚡ БЫСТРАЯ загрузка: {mesh_file_path}")
            
            # Быстрая загрузка
            if use_fastmesh:
                mesh_out = FastMesh.load(
                    mesh_file_path,
                    resize=True,
                    renormal=True,
                    retex=not skip_uv_generation,  # Пропускаем UV если нужно
                    clean=False  # Пропускаем clean для скорости
                )
                mesh_type = "FastMesh"
            else:
                mesh_out = Mesh.load(
                    mesh_file_path,
                    resize=True,
                    renormal=True,
                    retex=not skip_uv_generation,
                    clean=False,
                    use_new_gltf_loader=True  # Используем наш быстрый загрузчик
                )
                mesh_type = "Standard Mesh"
            
            if mesh_out is None:
                info = "❌ Не удалось загрузить mesh"
                return (None, info)
            
            # Убираем текстуры если нужно ускорить
            if skip_texture_loading:
                mesh_out.albedo = None
                mesh_out.metallicRoughness = None
                info += "\n⚡ Текстуры пропущены для скорости"
            
            # Информация
            info_lines = [
                f"✅ {mesh_type} загружен успешно",
                f"📊 Вершины: {mesh_out.v.shape[0]}",
                f"🔺 Грани: {mesh_out.f.shape[0]}",
            ]
            
            if mesh_out.vt is not None:
                info_lines.append(f"🗺️ UV: {mesh_out.vt.shape[0]}")
            else:
                info_lines.append("🗺️ UV: Нет")
            
            if mesh_out.albedo is not None:
                info_lines.append(f"🎨 Albedo: {mesh_out.albedo.shape}")
            else:
                info_lines.append("🎨 Albedo: Нет")
            
            info = "\n".join(info_lines)
            
            print(f"⚡ Быстрая загрузка завершена!")
            
        except Exception as e:
            info = f"❌ Ошибка загрузки: {e}"
            
        return (mesh_out, info)


# Mapping для ComfyUI
NODE_CLASS_MAPPINGS = {
    "Hunyuan3D_21_ShapeGen_Optimized": Hunyuan3D_21_ShapeGen_Optimized,
    "Hunyuan3D_21_TexGen_Optimized": Hunyuan3D_21_TexGen_Optimized,
    "FastMesh_Simple_Loader": FastMesh_Simple_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D_21_ShapeGen_Optimized": "Hunyuan3D 2.1 ShapeGen (FAST)",
    "Hunyuan3D_21_TexGen_Optimized": "Hunyuan3D 2.1 TexGen (FAST)",
    "FastMesh_Simple_Loader": "Fast Mesh Loader",
}
