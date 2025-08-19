#!/usr/bin/env python3
"""
Тест нового GLTF загрузчика mesh_processor
Сравнение с оригинальным trimesh-based загрузчиком
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Добавляем путь к mesh_processor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh import Mesh


def compare_loaders(glb_path: str):
    """Сравнивает старый и новый загрузчики GLB"""
    
    if not os.path.exists(glb_path):
        print(f"❌ Файл не найден: {glb_path}")
        return False
    
    print(f"🔍 Тестируем файл: {glb_path}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    # Тест старого загрузчика (trimesh)
    print("\n📦 Загрузка через trimesh (load_trimesh):")
    try:
        start_time = time.time()
        mesh_old = Mesh.load(glb_path, use_new_gltf_loader=False, device=device)
        old_time = time.time() - start_time
        
        if mesh_old is not None:
            print(f"✅ Загружен успешно за {old_time:.3f}s")
            print(f"   Вершины: {mesh_old.v.shape if mesh_old.v is not None else 'None'}")
            print(f"   Грани: {mesh_old.f.shape if mesh_old.f is not None else 'None'}")
            print(f"   UV: {mesh_old.vt.shape if mesh_old.vt is not None else 'None'}")
            print(f"   Нормали: {mesh_old.vn.shape if mesh_old.vn is not None else 'None'}")
            print(f"   Цвета вершин: {mesh_old.vc.shape if mesh_old.vc is not None else 'None'}")
            print(f"   Альбедо: {mesh_old.albedo.shape if mesh_old.albedo is not None else 'None'}")
        else:
            print("❌ Ошибка загрузки")
            return False
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False
    
    # Тест нового загрузчика (mesh_processor)
    print("\n🆕 Загрузка через mesh_processor (load_gltf):")
    try:
        start_time = time.time()
        mesh_new = Mesh.load(glb_path, use_new_gltf_loader=True, device=device)
        new_time = time.time() - start_time
        
        if mesh_new is not None:
            print(f"✅ Загружен успешно за {new_time:.3f}s")
            print(f"   Вершины: {mesh_new.v.shape if mesh_new.v is not None else 'None'}")
            print(f"   Грани: {mesh_new.f.shape if mesh_new.f is not None else 'None'}")
            print(f"   UV: {mesh_new.vt.shape if mesh_new.vt is not None else 'None'}")
            print(f"   Нормали: {mesh_new.vn.shape if mesh_new.vn is not None else 'None'}")
            print(f"   Цвета вершин: {mesh_new.vc.shape if mesh_new.vc is not None else 'None'}")
            print(f"   Альбедо: {mesh_new.albedo.shape if mesh_new.albedo is not None else 'None'}")
        else:
            print("❌ Ошибка загрузки")
            return False
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False
    
    # Сравнение результатов
    print("\n🔍 Сравнение результатов:")
    print(f"⏱️ Время: старый {old_time:.3f}s vs новый {new_time:.3f}s (разница: {((new_time/old_time - 1) * 100):+.1f}%)")
    
    # Сравнение геометрии
    if mesh_old.v is not None and mesh_new.v is not None:
        vertices_match = torch.allclose(mesh_old.v, mesh_new.v, atol=1e-5)
        print(f"📐 Вершины: {'✅ Совпадают' if vertices_match else '❌ Различаются'}")
        
        if not vertices_match:
            diff = torch.abs(mesh_old.v - mesh_new.v).max().item()
            print(f"   Максимальная разница: {diff:.6f}")
    
    if mesh_old.f is not None and mesh_new.f is not None:
        faces_match = torch.equal(mesh_old.f, mesh_new.f)
        print(f"🔺 Грани: {'✅ Совпадают' if faces_match else '❌ Различаются'}")
    
    # Сравнение UV
    if mesh_old.vt is not None and mesh_new.vt is not None:
        uv_match = torch.allclose(mesh_old.vt, mesh_new.vt, atol=1e-5)
        print(f"🗺️ UV: {'✅ Совпадают' if uv_match else '❌ Различаются'}")
    elif mesh_old.vt is None and mesh_new.vt is None:
        print("🗺️ UV: ✅ Оба отсутствуют")
    else:
        print("🗺️ UV: ❌ Один есть, другой отсутствует")
    
    # Сравнение текстур
    if mesh_old.albedo is not None and mesh_new.albedo is not None:
        texture_match = torch.allclose(mesh_old.albedo, mesh_new.albedo, atol=1e-3)
        print(f"🎨 Альбедо: {'✅ Совпадают' if texture_match else '❌ Различаются'}")
    elif mesh_old.albedo is None and mesh_new.albedo is None:
        print("🎨 Альбедо: ✅ Оба отсутствуют")
    else:
        print("🎨 Альбедо: ❌ Один есть, другой отсутствует")
    
    print("\n" + "=" * 60)
    return True


def main():
    print("🧪 Тест нового GLTF загрузчика mesh_processor")
    print("=" * 60)
    
    # Примеры файлов для тестирования
    test_files = [
        # Добавьте пути к вашим GLB файлам здесь
        "test.glb",
        "example.glb", 
        "model.glb"
    ]
    
    # Ищем файлы в текущей директории
    current_dir = Path(".")
    glb_files = list(current_dir.glob("*.glb")) + list(current_dir.glob("*.gltf"))
    
    if glb_files:
        print(f"Найдено {len(glb_files)} GLB/GLTF файлов:")
        for glb_file in glb_files:
            print(f"  - {glb_file}")
        print()
        
        for glb_file in glb_files[:3]:  # Тестируем первые 3 файла
            success = compare_loaders(str(glb_file))
            if not success:
                break
            print()
    else:
        print("❌ GLB/GLTF файлы не найдены в текущей директории")
        print("Создайте тестовые файлы или укажите пути в переменной test_files")


if __name__ == "__main__":
    main()