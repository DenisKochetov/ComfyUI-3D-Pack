# Hunyuan3DDiTFlowMatchingPipeline_2_1
# Hunyuan3DPaintPipeline_2_1

class Hunyuan3D_21_ShapeGen_New:
    """Hunyuan3D-2.1 Shape Generation с новым конвертером mesh"""
    
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
                "use_new_converter": ("BOOLEAN", {"default": True, "tooltip": "Использовать новый конвертер mesh (экспериментально)"}),
            }
        }

    @torch.no_grad()
    def generate(self, shapegen_pipe, image, seed, steps, guidance_scale, octree_resolution, remove_background, auto_cleanup, use_new_converter):
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
        
        face_reduce_worker = FaceReducer_2_1()
        mesh = face_reduce_worker(mesh)
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
        
        # Используем новый конвертер
        mesh_out = Mesh.from_trimesh(mesh, use_new_converter=use_new_converter)
        
        if mesh_out is not None:
            mesh_out.auto_normal()
            converter_type = "новый mesh_processor" if use_new_converter else "старый trimesh"
            print(f"[ShapeGen] Использован {converter_type} конвертер")
        else:
            print("[ShapeGen] Ошибка конвертации mesh, используем fallback")
            mesh_out = Mesh.load_trimesh(given_mesh=mesh)
            mesh_out.auto_normal()
        
        processed_image_tensor = pils_to_torch_imgs([pil_image])
        
        return (mesh_out, processed_image_tensor)


class Hunyuan3D_21_TexGen_New:
    """Hunyuan3D-2.1 Texture Generation с новым конвертером и поддержкой GLB загрузки"""
    
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
                "use_new_converter": ("BOOLEAN", {"default": True, "tooltip": "Использовать новый конвертер mesh"}),
                "use_new_gltf_loader": ("BOOLEAN", {"default": True, "tooltip": "Использовать новый GLB/GLTF загрузчик"}),
            }
        }

    @torch.no_grad()
    def generate(self, texgen_pipe, mesh_path, image, create_pbr, use_remesh, use_new_converter, use_new_gltf_loader):
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
                        print(f"Created GLB with full PBR materials: {glb_path}")
                    except Exception as e:
                        print(f"Warning: Failed to create GLB with PBR materials: {e}")
                        # Fallback to basic conversion
                        from .Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb
                        convert_obj_to_glb(result_path, glb_path)
                        print(f"Created GLB with basic conversion: {glb_path}")
                
                if os.path.exists(glb_path):
                    try:
                        # Попытка загрузки с новым загрузчиком
                        if use_new_gltf_loader:
                            try:
                                mesh_out = Mesh.load_gltf(glb_path)
                                if mesh_out is not None:
                                    mesh_out.auto_normal()
                                    print(f"✅ Загружен GLB новым загрузчиком: {glb_path}")
                                else:
                                    print("⚠️ Новый загрузчик вернул None, пробуем старый")
                                    raise Exception("New loader returned None")
                            except Exception as e:
                                print(f"⚠️ Ошибка нового загрузчика: {e}")
                                print("🔄 Fallback к старому загрузчику...")
                                use_new_gltf_loader = False
                        
                        # Fallback к старому способу через trimesh
                        if not use_new_gltf_loader or mesh_out is None:
                            glb_scene = trimesh.load(glb_path)
                            
                            if hasattr(glb_scene, 'geometry') and glb_scene.geometry:
                                mesh_name = list(glb_scene.geometry.keys())[0]
                                glb_mesh = glb_scene.geometry[mesh_name]
                            else:
                                glb_mesh = glb_scene
                            
                            mesh_out = Mesh.from_trimesh(glb_mesh, use_new_converter=use_new_converter)
                            
                            if mesh_out is not None:
                                mesh_out.auto_normal()
                                converter_type = "новый mesh_processor" if use_new_converter else "старый trimesh"
                                print(f"✅ Загружен GLB через trimesh + {converter_type}: {glb_path}")
                            else:
                                print("⚠️ Новый конвертер вернул None, используем старый")
                                mesh_out = Mesh.load_trimesh(given_mesh=glb_mesh)
                                mesh_out.auto_normal()
                                print(f"✅ Загружен GLB через старый способ: {glb_path}")
                                
                    except Exception as e:
                        print(f"❌ Ошибка загрузки GLB: {e}")
                        print(f"GLB path: {glb_path}")
            
            # If PBR failed or not requested, load regular textured mesh
            if mesh_out is None:
                try:
                    textured_mesh = trimesh.load(result_path)
                    mesh_out = Mesh.from_trimesh(textured_mesh, use_new_converter=use_new_converter)
                    
                    if mesh_out is not None:
                        mesh_out.auto_normal()
                        converter_type = "новый mesh_processor" if use_new_converter else "старый trimesh"
                        if create_pbr:
                            print(f"⚠️ PBR не удался, загружена обычная текстурированная модель ({converter_type})")
                        else:
                            print(f"✅ Загружена текстурированная модель ({converter_type})")
                    else:
                        print("⚠️ Новый конвертер вернул None, используем старый")
                        mesh_out = Mesh.load_trimesh(given_mesh=textured_mesh)
                        mesh_out.auto_normal()
                        print("✅ Загружена текстурированная модель (старый способ)")
                        
                except Exception as e:
                    print(f"❌ Критическая ошибка загрузки mesh: {e}")
                    raise e
            
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


class Hunyuan3D_21_Compare:
    """Сравнение старого и нового подходов для Hunyuan3D-2.1"""
    
    CATEGORY = "Comfy3D/Algorithm/Hunyuan3D-2.1"
    RETURN_TYPES = ("MESH", "MESH", "STRING")
    RETURN_NAMES = ("mesh_old", "mesh_new", "comparison_report")
    FUNCTION = "compare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh_object": ("MESH",),  # Принимаем готовый mesh от другой ноды
            }
        }

    def compare(self, trimesh_object):
        import time
        
        # Получаем исходный trimesh объект (предполагаем что это еще не конвертированный)
        # Для демонстрации создадим простой тест
        
        try:
            # Старый способ
            start_time = time.time()
            mesh_old = Mesh.load_trimesh(given_mesh=trimesh_object)
            mesh_old.auto_normal()
            old_time = time.time() - start_time
            old_status = f"✅ {old_time:.3f}s"
        except Exception as e:
            mesh_old = None
            old_status = f"❌ {str(e)[:50]}"
            old_time = 0
        
        try:
            # Новый способ  
            start_time = time.time()
            mesh_new = Mesh.from_trimesh(trimesh_object, use_new_converter=True)
            if mesh_new is not None:
                mesh_new.auto_normal()
            new_time = time.time() - start_time
            new_status = f"✅ {new_time:.3f}s"
        except Exception as e:
            mesh_new = None
            new_status = f"❌ {str(e)[:50]}"
            new_time = 0
        
        # Создаем отчет
        report_lines = [
            "📊 Сравнение конверторов Hunyuan3D-2.1",
            f"🕰️ Старый (load_trimesh): {old_status}",
            f"🆕 Новый (from_trimesh): {new_status}",
        ]
        
        if mesh_old and mesh_new and old_time > 0 and new_time > 0:
            speed_diff = ((new_time / old_time - 1) * 100)
            report_lines.append(f"⚡ Скорость: {speed_diff:+.1f}%")
            
            # Сравнение геометрии
            if mesh_old.v is not None and mesh_new.v is not None:
                vertices_match = torch.allclose(mesh_old.v, mesh_new.v, atol=1e-5)
                report_lines.append(f"📐 Вершины: {'✅' if vertices_match else '❌'}")
            
            if mesh_old.f is not None and mesh_new.f is not None:
                faces_match = torch.equal(mesh_old.f, mesh_new.f)
                report_lines.append(f"🔺 Грани: {'✅' if faces_match else '❌'}")
            
            # UV и текстуры
            uv_old = mesh_old.vt is not None
            uv_new = mesh_new.vt is not None
            report_lines.append(f"🗺️ UV: старый={uv_old}, новый={uv_new}")
            
            tex_old = mesh_old.albedo is not None
            tex_new = mesh_new.albedo is not None
            report_lines.append(f"🎨 Текстуры: старый={tex_old}, новый={tex_new}")
        
        report = "\n".join(report_lines)
        return (mesh_old, mesh_new, report)