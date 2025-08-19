# Hunyuan3D-2.1 с FastMesh

import os
import gc
import torch


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
                        create_glb_with_pbr_materials_2_1(result_path, textures_dict, glb_path)
                        print(f"✅ Создан GLB с PBR материалами: {glb_path}")
                    except Exception as e:
                        print(f"⚠️ Ошибка создания GLB с PBR: {e}")
                        # Fallback to basic conversion
                        from .Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb
                        convert_obj_to_glb(result_path, glb_path)
                        print(f"✅ Создан GLB базовой конвертацией: {glb_path}")
                
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
        
        if output_fastmesh:
            try:
                print("🚀 Попытка конвертации в FastMesh...")
                # Сохраняем во временный файл и загружаем через FastMesh
                temp_glb_path = "temp_shapegen_fastmesh.glb"
                trimesh_obj.export(temp_glb_path)
                
                try:
                    mesh_out = FastMesh.load(temp_glb_path)
                    
                    if mesh_out is not None:
                        print("✅ Конвертация в FastMesh успешна!")
                    else:
                        print("⚠️ FastMesh конвертация вернула None")
                        
                finally:
                    # Удаляем временный файл
                    if os.path.exists(temp_glb_path):
                        os.remove(temp_glb_path)
                        
            except Exception as e:
                print(f"❌ Ошибка FastMesh конвертации: {e}")
        
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