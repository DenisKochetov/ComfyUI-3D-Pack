# Hunyuan3D-2.1 с FastGLB вместо trimesh

import os
import gc
import torch
from mesh_processor import fastglb  # Наша новая библиотека


class Hunyuan3D_21_TexGen_FastGLB:
    """Hunyuan3D-2.1 Texture Generation с FastGLB (без trimesh)"""
    
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
            }
        }

    @torch.no_grad()
    def generate(self, texgen_pipe, mesh_path, image, create_pbr, use_remesh):
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
                        create_glb_with_pbr_materials_2_1(result_path, textures_dict, glb_path)
                        print(f"✅ Создан GLB с PBR материалами: {glb_path}")
                    except Exception as e:
                        print(f"⚠️ Ошибка создания GLB с PBR: {e}")
                        # Fallback to basic conversion
                        from .Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb
                        convert_obj_to_glb(result_path, glb_path)
                        print(f"✅ Создан GLB базовой конвертацией: {glb_path}")
                
                # Загружаем GLB через FastGLB
                if os.path.exists(glb_path):
                    try:
                        print(f"🚀 Загружаем GLB через FastGLB: {glb_path}")
                        fastglb_mesh = fastglb.load(glb_path)
                        
                        # Конвертируем FastGLB в наш Mesh объект
                        mesh_out = fastglb_to_mesh(fastglb_mesh)
                        
                        if mesh_out is not None:
                            mesh_out.auto_normal()
                            print(f"✅ GLB загружен успешно через FastGLB!")
                            print(f"   Вершины: {mesh_out.v.shape}")
                            print(f"   Грани: {mesh_out.f.shape}")
                            print(f"   UV: {mesh_out.vt.shape if mesh_out.vt is not None else 'None'}")
                            print(f"   Текстуры: {mesh_out.albedo.shape if mesh_out.albedo is not None else 'None'}")
                        else:
                            print("❌ FastGLB конвертация не удалась")
                            
                    except Exception as e:
                        print(f"❌ Ошибка загрузки GLB через FastGLB: {e}")
                        mesh_out = None
            
            # Если GLB не удался, загружаем OBJ через стандартный загрузчик
            if mesh_out is None:
                try:
                    print(f"🔄 Fallback: загружаем OBJ: {result_path}")
                    mesh_out = Mesh.load_obj(result_path)
                    
                    if mesh_out is not None:
                        mesh_out.auto_normal()
                        if create_pbr:
                            print("⚠️ PBR GLB не удался, загружена OBJ модель")
                        else:
                            print("✅ Загружена OBJ модель")
                    else:
                        raise Exception("OBJ загрузка не удалась")
                        
                except Exception as e:
                    print(f"❌ Критическая ошибка: {e}")
                    # Последний fallback - через trimesh (если он всё еще доступен)
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


class Hunyuan3D_21_ShapeGen_FastGLB:
    """Hunyuan3D-2.1 Shape Generation с FastGLB конвертацией"""
    
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
                "try_fastglb_conversion": ("BOOLEAN", {"default": True, "tooltip": "Попытаться конвертировать через FastGLB"}),
            }
        }

    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, remove_background, auto_cleanup, try_fastglb_conversion):
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
        
        # Получаем trimesh объект из пайплайна
        trimesh_obj = export_to_trimesh_2_1(outputs)[0]
        
        face_reduce_worker = FaceReducer_2_1()
        trimesh_obj = face_reduce_worker(trimesh_obj)
        del face_reduce_worker
        
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
        
        # Конвертируем trimesh в наш Mesh объект
        mesh_out = None
        
        if try_fastglb_conversion:
            try:
                print("🚀 Попытка конвертации через FastGLB...")
                # Сохраняем во временный GLB файл и загружаем через FastGLB
                temp_glb_path = "temp_fastglb_test.glb"
                trimesh_obj.export(temp_glb_path)
                
                try:
                    fastglb_mesh = fastglb.load(temp_glb_path)
                    mesh_out = fastglb_to_mesh(fastglb_mesh)
                    
                    if mesh_out is not None:
                        mesh_out.auto_normal()
                        print("✅ Конвертация через FastGLB успешна!")
                    else:
                        print("⚠️ FastGLB конвертация вернула None")
                        
                finally:
                    # Удаляем временный файл
                    if os.path.exists(temp_glb_path):
                        os.remove(temp_glb_path)
                        
            except Exception as e:
                print(f"❌ Ошибка FastGLB конвертации: {e}")
        
        # Fallback к стандартному методу
        if mesh_out is None:
            try:
                print("🔄 Fallback к стандартной конвертации...")
                mesh_out = Mesh.from_trimesh(trimesh_obj, use_new_converter=True)
                
                if mesh_out is not None:
                    mesh_out.auto_normal()
                    print("✅ Стандартная конвертация успешна")
                else:
                    print("⚠️ Стандартная конвертация вернула None, используем старую")
                    mesh_out = Mesh.load_trimesh(given_mesh=trimesh_obj)
                    mesh_out.auto_normal()
                    
            except Exception as e:
                print(f"❌ Ошибка стандартной конвертации: {e}")
                print("🆘 Последний fallback...")
                mesh_out = Mesh.load_trimesh(given_mesh=trimesh_obj)
                mesh_out.auto_normal()
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        return (mesh_out, processed_image_tensor)


def fastglb_to_mesh(fastglb_mesh) -> 'Mesh':
    """Конвертирует FastGLB объект в наш Mesh объект"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Создаем новый Mesh объект
    mesh = Mesh()
    mesh.device = device
    
    # Основная геометрия
    if len(fastglb_mesh.vertices) > 0:
        mesh.v = torch.tensor(fastglb_mesh.vertices, dtype=torch.float32, device=device)
    else:
        print("⚠️ FastGLB: нет вершин")
        return None
    
    if len(fastglb_mesh.faces) > 0:
        mesh.f = torch.tensor(fastglb_mesh.faces, dtype=torch.int32, device=device)
    else:
        print("⚠️ FastGLB: нет граней")
        return None
    
    # UV координаты
    if fastglb_mesh.uv_coordinates is not None:
        mesh.vt = torch.tensor(fastglb_mesh.uv_coordinates, dtype=torch.float32, device=device)
        mesh.ft = mesh.f  # Используем те же индексы
        print(f"FastGLB: UV координаты {mesh.vt.shape}")
    
    # Нормали
    if fastglb_mesh.vertex_normals is not None:
        mesh.vn = torch.tensor(fastglb_mesh.vertex_normals, dtype=torch.float32, device=device)
        mesh.fn = mesh.f  # Используем те же индексы
        print(f"FastGLB: Нормали {mesh.vn.shape}")
    
    # Цвета вершин
    if fastglb_mesh.vertex_colors is not None:
        mesh.vc = torch.tensor(fastglb_mesh.vertex_colors, dtype=torch.float32, device=device)
        print(f"FastGLB: Цвета вершин {mesh.vc.shape}")
    
    # Текстуры
    should_create_empty_albedo = True
    
    if 'albedo' in fastglb_mesh.textures:
        try:
            albedo_texture = fastglb_mesh.textures['albedo']
            if albedo_texture is not None:
                albedo_float = albedo_texture.astype(np.float32) / 255.0
                mesh.albedo = torch.tensor(albedo_float, dtype=torch.float32, device=device).contiguous()
                should_create_empty_albedo = False
                print(f"FastGLB: Albedo текстура {mesh.albedo.shape}")
        except Exception as e:
            print(f"FastGLB: Ошибка загрузки albedo: {e}")
    
    if 'metallicRoughness' in fastglb_mesh.textures:
        try:
            mr_texture = fastglb_mesh.textures['metallicRoughness']
            if mr_texture is not None:
                mr_float = mr_texture.astype(np.float32) / 255.0
                mesh.metallicRoughness = torch.tensor(mr_float, dtype=torch.float32, device=device).contiguous()
                print(f"FastGLB: MetallicRoughness текстура {mesh.metallicRoughness.shape}")
        except Exception as e:
            print(f"FastGLB: Ошибка загрузки metallicRoughness: {e}")
    
    # Создаем пустую текстуру если нужно
    if should_create_empty_albedo:
        mesh.set_new_albedo(1024, 1024)
        print(f"FastGLB: Создана пустая текстура {mesh.albedo.shape}")
    
    return mesh


# Функция-замена для trimesh.load в существующем коде
def fastglb_load_replacement(path):
    """Замена trimesh.load() для GLB файлов"""
    
    if path.lower().endswith(('.glb', '.gltf')):
        print(f"🚀 FastGLB загрузка: {path}")
        return fastglb.load(path)
    else:
        # Для других форматов используем оригинальный trimesh
        print(f"⚠️ Не GLB/GLTF, используем trimesh: {path}")
        import trimesh
        return trimesh.load(path)