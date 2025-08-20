# Hunyuan3D-2.1 с FastMesh

import os
import gc
import torch
import numpy as np
from mesh_processor.export_utils import export_to_fastmesh, export_to_mesh
from mesh_processor.mesh import FastMesh, Mesh

# Импорты для работы с изображениями и пайплайнами
try:
    from shared_utils.image_utils import torch_imgs_to_pils, pils_to_torch_imgs
    from Gen_3D_Modules.Hunyuan3D_2_1 import (
        BackgroundRemover_2_1,
        FaceReducer_2_1,
        export_to_trimesh_2_1,
        create_glb_with_pbr_materials_2_1
    )
except ImportError as e:
    print(f"⚠️ Некоторые модули недоступны: {e}")
    # Создаем заглушки
    def torch_imgs_to_pils(imgs): return []
    def pils_to_torch_imgs(pils): return torch.tensor([])
    def BackgroundRemover_2_1(): return None
    def FaceReducer_2_1(): return None
    def export_to_trimesh_2_1(x): return [None]
    def create_glb_with_pbr_materials_2_1(*args): return None


class Hunyuan3D_21_TexGen_FastMesh:
    """Hunyuan3D-2.1 Texture Generation с FastMesh (без trimesh для GLB)"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("textured_mesh",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texgen_pipe": ("DIFFUSERS_PIPE",),
                "mesh_path": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "create_pbr": ("BOOLEAN", {"default": True}),
                "use_remesh": ("BOOLEAN", {"default": False}),
                "use_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Использовать FastMesh для GLB загрузки"}),
            }
        }

    @torch.no_grad()
    def generate(self, texgen_pipe, mesh_path, image, create_pbr, use_remesh, use_fastmesh):
        if not mesh_path or not os.path.exists(mesh_path):
            raise Exception(f"Mesh file not found: {mesh_path}")

        pil_image = torch_imgs_to_pils(image)[0]
        
        # Save files to output/Hun2-1 directory
        output_dir = "output/Hun2-1"
        os.makedirs(output_dir, exist_ok=True)
        
        image_path = os.path.join(output_dir, "hunyuan_input.png")
        output_path = os.path.join(output_dir, "hunyuan_output.obj")
        
        try:
            pil_image.save(image_path)
            
            result_path = texgen_pipe(
                mesh_path=mesh_path,
                image_path=image_path,
                output_mesh_path=output_path,
                save_glb=True,  # Always create GLB
                use_remesh=use_remesh
            )
            
            mesh_out = None
            
            if create_pbr:
                glb_path = result_path.replace(".obj", ".glb")
                
                if not os.path.exists(glb_path):
                    base_path = os.path.splitext(result_path)[0]
                    textures_dict = {
                        'albedo': f"{base_path}.jpg",
                    }
                    
                    metallic_path = f"{base_path}_metallic.jpg"
                    roughness_path = f"{base_path}_roughness.jpg"
                    normal_path = f"{base_path}_normal.jpg"
                    
                    if os.path.exists(metallic_path):
                        textures_dict['metallic'] = metallic_path
                    if os.path.exists(roughness_path):
                        textures_dict['roughness'] = roughness_path
                    if os.path.exists(normal_path):
                        textures_dict['normal'] = normal_path
                    
                    try:
                        # Используем нашу собственную функцию для создания GLB
                        print("🎨 Создаем GLB с нашей собственной функцией...")
                        self._create_glb_from_obj_and_textures(result_path, textures_dict, glb_path)
                        print(f"✅ Создан GLB с нашей функцией: {glb_path}")
                    except Exception as e:
                        print(f"⚠️ Ошибка создания GLB: {e}")
                        # Fallback к оригинальной функции
                        try:
                            create_glb_with_pbr_materials_2_1(result_path, textures_dict, glb_path)
                            print(f"✅ Создан GLB с PBR материалами (fallback): {glb_path}")
                        except Exception as e2:
                            print(f"⚠️ Fallback также failed: {e2}")
                            # Последний fallback - простая конвертация OBJ
                            if use_fastmesh:
                                temp_mesh = FastMesh.load(result_path)
                            else:
                                temp_mesh = Mesh.load_obj(result_path)
                            if temp_mesh is not None:
                                temp_mesh.write(glb_path)
                                print(f"✅ Создан GLB простой конвертацией: {glb_path}")
                
                # Загружаем GLB через FastMesh или стандартный Mesh
                if os.path.exists(glb_path):
                    try:
                        if use_fastmesh:
                            print(f"🚀 Загружаем GLB через FastMesh: {glb_path}")
                            mesh_out = FastMesh.load(glb_path)
                            
                            if mesh_out is not None:
                                print(f"✅ FastMesh загрузка успешна!")
                                print(f"   Вершины: {mesh_out.v.shape}")
                                print(f"   Грани: {mesh_out.f.shape}")
                                print(f"   UV: {mesh_out.vt.shape if mesh_out.vt is not None else 'None'}")
                                print(f"   Текстуры: {mesh_out.albedo.shape if mesh_out.albedo is not None else 'None'}")
                            else:
                                print("❌ FastMesh вернул None")
                                use_fastmesh = False
                        
                        # Fallback к стандартному способу
                        if not use_fastmesh or mesh_out is None:
                            print(f"🔄 Fallback к стандартному Mesh.load_gltf: {glb_path}")
                            mesh_out = Mesh.load_gltf(glb_path)
                            
                            if mesh_out is not None:
                                mesh_out.auto_normal()
                                print(f"✅ Стандартная загрузка GLB успешна")
                            else:
                                print("⚠️ Стандартная загрузка тоже не удалась")
                                
                    except Exception as e:
                        print(f"❌ Ошибка загрузки GLB: {e}")
                        mesh_out = None
            
            # Если GLB не удался, загружаем OBJ
            if mesh_out is None:
                try:
                    if use_fastmesh:
                        print(f"🔄 FastMesh fallback к OBJ: {result_path}")
                        mesh_out = FastMesh.load(result_path)
                    else:
                        print(f"🔄 Стандартная загрузка OBJ: {result_path}")
                        mesh_out = Mesh.load_obj(result_path)
                    
                    if mesh_out is not None:
                        if not hasattr(mesh_out, 'auto_normal') or mesh_out.auto_normal is None:
                            # Если это FastMesh, у него уже есть auto_normal
                            pass
                        else:
                            mesh_out.auto_normal()
                        
                        if create_pbr:
                            print("⚠️ PBR GLB не удался, загружена OBJ модель")
                        else:
                            print("✅ Загружена OBJ модель")
                    else:
                        raise Exception("OBJ загрузка не удалась")
                        
                except Exception as e:
                    print(f"❌ Критическая ошибка: {e}")
                    # Последний fallback - через trimesh
                    print("🆘 Последний fallback через trimesh...")
                    import trimesh
                    textured_mesh = trimesh.load(result_path)
                    mesh_out = Mesh.load_trimesh(given_mesh=textured_mesh)
                    mesh_out.auto_normal()
                    print("✅ Загружено через trimesh (fallback)")
            
            return (mesh_out,)
            
        finally:
            # Clean up files
            torch.cuda.empty_cache()
            gc.collect()
            
            for file_path in [image_path, output_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass


class Hunyuan3D_21_ShapeGen_FastMesh:
    """Hunyuan3D-2.1 Shape Generation с поддержкой FastMesh"""
    
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
                "output_fastmesh": ("BOOLEAN", {"default": False, "tooltip": "Выводить FastMesh вместо стандартного Mesh"}),
            }
        }

    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, remove_background, auto_cleanup, output_fastmesh):
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
        
        # Получаем mesh объект НАПРЯМУЮ из пайплайна (БЕЗ trimesh!)
        if output_fastmesh:
            print("🚀 Экспорт из пайплайна напрямую в FastMesh...")
            mesh_objects = export_to_fastmesh(outputs)
            mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
        else:
            print("🔧 Экспорт из пайплайна в стандартный Mesh...")
            mesh_objects = export_to_mesh(outputs)
            mesh_out = mesh_objects[0] if isinstance(mesh_objects, list) else mesh_objects
        
        # Face reduction (используем нашу собственную реализацию)
        if mesh_out is not None:
            try:
                print("🔧 Применяем face reduction (собственная реализация)...")
                mesh_out = self._apply_face_reduction(mesh_out)
                print("✅ Face reduction применен успешно")
            except Exception as e:
                print(f"⚠️ Ошибка face reduction: {e}, пропускаем")
        
        # Больше НЕ нужна конвертация trimesh_obj!
        
        # Auto cleanup pipeline if enabled
        if auto_cleanup:
            try:
                shapegen_pipe.to('cpu')
                if hasattr(shapegen_pipe, 'unet'):
                    del shapegen_pipe.unet
                if hasattr(shapegen_pipe, 'vae'):
                    del shapegen_pipe.vae
                if hasattr(shapegen_pipe, 'scheduler'):
                    del shapegen_pipe.scheduler
                del outputs
                torch.cuda.empty_cache()
                gc.collect()
                print("Shape pipeline cleaned up")
            except Exception as e:
                print(f"Error during pipeline cleanup: {e}")
        
        # mesh_out уже создан выше напрямую из пайплайна!
        if mesh_out is None:
            print("❌ Критическая ошибка: не удалось получить mesh из пайплайна")
            return (None, processed_image_tensor)
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        return (mesh_out, processed_image_tensor)
    
    def _apply_face_reduction(self, mesh_obj):
        """Собственная реализация face reduction"""
        try:
            # Простая стратегия - удаляем каждый второй треугольник с маленькой площадью
            vertices = mesh_obj.v
            faces = mesh_obj.f
            
            # Вычисляем площади треугольников
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]] 
            v2 = vertices[faces[:, 2]]
            
            # Векторы сторон
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # Площадь через векторное произведение
            cross_product = torch.cross(edge1, edge2, dim=1)
            areas = 0.5 * torch.norm(cross_product, dim=1)
            
            # Оставляем треугольники с площадью больше медианы
            median_area = torch.median(areas)
            keep_mask = areas >= median_area * 0.5  # Более мягкий фильтр
            
            if keep_mask.sum() > faces.shape[0] * 0.3:  # Сохраняем минимум 30%
                new_faces = faces[keep_mask]
                mesh_obj.f = new_faces
                
                # Обновляем fn если есть
                if mesh_obj.fn is not None:
                    mesh_obj.fn = new_faces
                    
                print(f"Face reduction: {faces.shape[0]} → {new_faces.shape[0]} граней")
            else:
                print("Face reduction пропущен - слишком агрессивная фильтрация")
                
        except Exception as e:
            print(f"Ошибка в face reduction: {e}")
            
        return mesh_obj
    
    def _create_glb_with_textures(self, mesh_obj, texture_image, output_path, create_pbr=True):
        """Создает GLB файл с текстурами используя наш mesh_obj"""
        try:
            print(f"🎨 Создаем GLB с текстурами: {output_path}")
            
            # Конвертируем PIL в tensor если нужно
            if hasattr(texture_image, 'convert'):
                # PIL Image
                import numpy as np
                texture_array = np.array(texture_image.convert('RGB')).astype(np.float32) / 255.0
                mesh_obj.albedo = torch.tensor(texture_array, dtype=torch.float32, device=mesh_obj.device)
            elif isinstance(texture_image, torch.Tensor):
                # Torch tensor
                if texture_image.dtype == torch.uint8:
                    mesh_obj.albedo = texture_image.float() / 255.0
                else:
                    mesh_obj.albedo = texture_image
            
            # Генерируем UV если их нет
            if mesh_obj.vt is None:
                print("🗺️ Генерируем UV координаты...")
                mesh_obj.auto_uv(cache_path=output_path)
            
            # Создаем metallic-roughness текстуру если нужно PBR
            if create_pbr and mesh_obj.metallicRoughness is None:
                print("⚙️ Создаем PBR текстуру...")
                # Создаем базовую metallic-roughness (серая = не металл, шероховатый)
                h, w = mesh_obj.albedo.shape[:2]
                pbr_texture = torch.zeros((h, w, 3), dtype=torch.float32, device=mesh_obj.device)
                pbr_texture[:, :, 1] = 0.8  # Roughness (зеленый канал)
                pbr_texture[:, :, 2] = 0.1  # Metallic (синий канал)
                mesh_obj.metallicRoughness = pbr_texture
            
            # Сохраняем GLB
            mesh_obj.write(output_path)
            print(f"✅ GLB создан: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания GLB: {e}")
            return False
    
    def _create_glb_from_obj_and_textures(self, obj_path, textures_dict, output_glb_path):
        """Создает GLB из OBJ файла и словаря текстур"""
        
        try:
            print(f"🔧 Загружаем OBJ: {obj_path}")
            
            # Загружаем OBJ как FastMesh или стандартный Mesh
            if hasattr(self, 'use_fastmesh') and self.use_fastmesh:
                mesh = FastMesh.load(obj_path)
            else:
                mesh = Mesh.load_obj(obj_path)
            
            if mesh is None:
                raise Exception("Не удалось загрузить OBJ файл")
            
            print(f"✅ OBJ загружен: {mesh.v.shape[0]} вершин, {mesh.f.shape[0]} граней")
            
            # Загружаем текстуры
            if 'albedo' in textures_dict and os.path.exists(textures_dict['albedo']):
                print(f"🎨 Загружаем albedo: {textures_dict['albedo']}")
                import cv2
                albedo = cv2.imread(textures_dict['albedo'], cv2.IMREAD_COLOR)
                if albedo is not None:
                    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                    albedo = albedo.astype(np.float32) / 255.0
                    mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=mesh.device)
                    print(f"✅ Albedo загружен: {albedo.shape}")
            
            # Создаем metallic-roughness текстуру из отдельных файлов
            metallic_data = None
            roughness_data = None
            
            if 'metallic' in textures_dict and os.path.exists(textures_dict['metallic']):
                print(f"⚙️ Загружаем metallic: {textures_dict['metallic']}")
                metallic = cv2.imread(textures_dict['metallic'], cv2.IMREAD_GRAYSCALE)
                if metallic is not None:
                    metallic_data = metallic.astype(np.float32) / 255.0
            
            if 'roughness' in textures_dict and os.path.exists(textures_dict['roughness']):
                print(f"🔧 Загружаем roughness: {textures_dict['roughness']}")
                roughness = cv2.imread(textures_dict['roughness'], cv2.IMREAD_GRAYSCALE)
                if roughness is not None:
                    roughness_data = roughness.astype(np.float32) / 255.0
            
            # Объединяем в metallic-roughness текстуру
            if metallic_data is not None and roughness_data is not None:
                h, w = metallic_data.shape
                mr_texture = np.zeros((h, w, 3), dtype=np.float32)
                mr_texture[:, :, 1] = roughness_data  # Green = roughness
                mr_texture[:, :, 2] = metallic_data   # Blue = metallic
                mesh.metallicRoughness = torch.tensor(mr_texture, dtype=torch.float32, device=mesh.device)
                print(f"✅ MetallicRoughness создан: {mr_texture.shape}")
            elif mesh.albedo is not None:
                # Создаем базовую metallic-roughness
                h, w = mesh.albedo.shape[:2]
                mr_texture = np.zeros((h, w, 3), dtype=np.float32)
                mr_texture[:, :, 1] = 0.8  # Medium roughness
                mr_texture[:, :, 2] = 0.1  # Low metallic
                mesh.metallicRoughness = torch.tensor(mr_texture, dtype=torch.float32, device=mesh.device)
                print("✅ Базовая MetallicRoughness создана")
            
            # Генерируем UV если их нет
            if mesh.vt is None and mesh.albedo is not None:
                print("🗺️ Генерируем UV координаты...")
                mesh.auto_uv(cache_path=obj_path)
            
            # Сохраняем GLB
            mesh.write(output_glb_path)
            print(f"✅ GLB сохранен: {output_glb_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания GLB из OBJ: {e}")
            return False


class FastMesh_Test:
    """Тест FastMesh vs стандартный Mesh"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("MESH", "MESH", "STRING")
    RETURN_NAMES = ("mesh_standard", "mesh_fast", "comparison_report")
    FUNCTION = "test"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": ""}),
            }
        }

    def test(self, glb_path):
        import time
        
        if not glb_path or not os.path.exists(glb_path):
            return (None, None, f"❌ Файл не найден: {glb_path}")
        
        if not glb_path.lower().endswith(('.glb', '.gltf')):
            return (None, None, f"❌ Не GLB/GLTF файл: {glb_path}")
        
        mesh_standard = None
        mesh_fast = None
        
        # Тест стандартного загрузчика
        try:
            start_time = time.time()
            mesh_standard = Mesh.load_gltf(glb_path)
            standard_time = time.time() - start_time
            standard_status = f"✅ {standard_time:.3f}s"
        except Exception as e:
            standard_status = f"❌ {str(e)[:50]}"
            standard_time = 0
        
        # Тест FastMesh
        try:
            start_time = time.time()
            mesh_fast = FastMesh.load(glb_path)
            fast_time = time.time() - start_time
            fast_status = f"✅ {fast_time:.3f}s"
        except Exception as e:
            fast_status = f"❌ {str(e)[:50]}"
            fast_time = 0
        
        # Создаем отчет
        report_lines = [
            f"📊 Сравнение загрузчиков GLB: {os.path.basename(glb_path)}",
            f"🔧 Стандартный Mesh.load_gltf: {standard_status}",
            f"🚀 FastMesh.load: {fast_status}",
        ]
        
        if mesh_standard and mesh_fast and standard_time > 0 and fast_time > 0:
            speed_diff = ((fast_time / standard_time - 1) * 100)
            report_lines.append(f"⚡ Скорость FastMesh: {speed_diff:+.1f}%")
            
            # Сравнение геометрии
            if mesh_standard.v is not None and mesh_fast.v is not None:
                vertices_match = torch.allclose(mesh_standard.v, mesh_fast.v, atol=1e-5)
                report_lines.append(f"📐 Вершины: {'✅ совпадают' if vertices_match else '❌ различаются'}")
            
            if mesh_standard.f is not None and mesh_fast.f is not None:
                faces_match = torch.equal(mesh_standard.f, mesh_fast.f)
                report_lines.append(f"🔺 Грани: {'✅ совпадают' if faces_match else '❌ различаются'}")
            
            # UV и текстуры
            uv_std = mesh_standard.vt is not None
            uv_fast = mesh_fast.vt is not None
            report_lines.append(f"🗺️ UV: стандартный={uv_std}, FastMesh={uv_fast}")
            
            tex_std = mesh_standard.albedo is not None
            tex_fast = mesh_fast.albedo is not None
            report_lines.append(f"🎨 Текстуры: стандартный={tex_std}, FastMesh={tex_fast}")
            
            if tex_std and tex_fast:
                try:
                    texture_match = torch.allclose(mesh_standard.albedo, mesh_fast.albedo, atol=1e-3)
                    report_lines.append(f"🖼️ Качество текстур: {'✅ идентичны' if texture_match else '❌ различаются'}")
                except:
                    report_lines.append("🖼️ Качество текстур: ❓ не удалось сравнить")
        
        report = "\n".join(report_lines)
        return (mesh_standard, mesh_fast, report)