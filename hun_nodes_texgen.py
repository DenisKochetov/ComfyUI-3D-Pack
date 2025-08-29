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
