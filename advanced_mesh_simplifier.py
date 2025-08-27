# Advanced Mesh Simplification - Quadric Edge Collapse для FastMesh/Mesh
# Реализация профессионального алгоритма упрощения мешей

import torch
import numpy as np
import heapq
from typing import Union, List, Dict, Set, Tuple, Optional
from mesh_processor.mesh import Mesh, FastMesh


class EdgeCollapseSimplifier:
    """
    Профессиональный упроститель мешей на основе Quadric Edge Collapse
    Аналог pymeshlab.meshing_decimation_quadric_edge_collapse
    """
    
    def __init__(self, preserve_boundary: bool = True, preserve_topology: bool = True):
        self.preserve_boundary = preserve_boundary
        self.preserve_topology = preserve_topology
        self.OPTIM_VALENCE = 6
        self.VALENCE_WEIGHT = 1.0
    
    def simplify(self, mesh_obj: Union[Mesh, FastMesh], target_faces: int) -> Union[Mesh, FastMesh]:
        """
        Основная функция упрощения меша
        
        Args:
            mesh_obj: FastMesh или Mesh объект
            target_faces: целевое количество граней
            
        Returns:
            Упрощенный mesh объект
        """
        print(f"[EdgeCollapseSimplifier] Начинаем упрощение: {mesh_obj.f.shape[0]} → {target_faces} граней")
        
        if mesh_obj.f.shape[0] <= target_faces:
            print(f"[EdgeCollapseSimplifier] Mesh уже имеет {mesh_obj.f.shape[0]} граней (target: {target_faces})")
            return mesh_obj
        
        # Конвертируем в numpy для удобства
        vertices = mesh_obj.v.detach().cpu().numpy().astype(np.float64)
        faces = mesh_obj.f.detach().cpu().numpy().astype(np.int32)
        
        # Строим топологию
        topology = self._build_topology(vertices, faces)
        
        # Вычисляем Q-матрицы для каждой вершины
        Q_matrices = self._compute_quadrics(vertices, faces, topology)
        
        # Строим приоритетную очередь ребер
        edge_heap = self._build_edge_heap(vertices, faces, topology, Q_matrices)
        
        # Выполняем схлопывание ребер
        simplified_vertices, simplified_faces = self._collapse_edges(
            vertices, faces, topology, Q_matrices, edge_heap, target_faces
        )
        
        # Обновляем mesh объект
        device = mesh_obj.v.device
        mesh_obj.v = torch.tensor(simplified_vertices, dtype=torch.float32, device=device)
        mesh_obj.f = torch.tensor(simplified_faces, dtype=torch.int32, device=device)
        
        # Пересчитываем нормали
        mesh_obj.auto_normal()
        if hasattr(mesh_obj, 'fn'):
            mesh_obj.fn = mesh_obj.f
        
        print(f"[EdgeCollapseSimplifier] Упрощение завершено: {mesh_obj.f.shape[0]} граней")
        return mesh_obj
    
    def _build_topology(self, vertices: np.ndarray, faces: np.ndarray) -> Dict:
        """Строим топологическую информацию меша"""
        
        num_vertices = len(vertices)
        num_faces = len(faces)
        
        # Vertex to faces
        vf = [set() for _ in range(num_vertices)]
        for face_id, face in enumerate(faces):
            vf[face[0]].add(face_id)
            vf[face[1]].add(face_id)
            vf[face[2]].add(face_id)
        
        # Edges
        edges = set()
        edge_to_faces = {}
        
        for face_id, face in enumerate(faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                edges.add(edge)
                
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_id)
        
        edges = list(edges)
        
        # Vertex to vertex connectivity
        v2v = [set() for _ in range(num_vertices)]
        for edge in edges:
            v1, v2 = edge
            v2v[v1].add(v2)
            v2v[v2].add(v1)
        
        return {
            'vf': vf,
            'v2v': v2v,
            'edges': edges,
            'edge_to_faces': edge_to_faces,
            'num_vertices': num_vertices,
            'num_faces': num_faces
        }
    
    def _compute_quadrics(self, vertices: np.ndarray, faces: np.ndarray, topology: Dict) -> List[np.ndarray]:
        """Вычисляем Q-матрицы для каждой вершины"""
        
        num_vertices = topology['num_vertices']
        vf = topology['vf']
        
        # Вычисляем нормали и центры граней
        face_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]], 
            vertices[faces[:, 2]] - vertices[faces[:, 0]]
        )
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
        
        face_centers = np.mean(vertices[faces], axis=1)
        
        # Q-матрицы для каждой вершины
        Q_matrices = []
        
        for v_id in range(num_vertices):
            Q = np.zeros((4, 4), dtype=np.float64)
            
            # Собираем грани этой вершины
            vertex_faces = list(vf[v_id])
            
            for face_id in vertex_faces:
                # Плоскость грани: ax + by + cz + d = 0
                normal = face_normals[face_id]
                center = face_centers[face_id]
                d = -np.dot(normal, center)
                
                # Коэффициенты плоскости [a, b, c, d]
                plane = np.array([normal[0], normal[1], normal[2], d])
                
                # Q += plane^T * plane
                Q += np.outer(plane, plane)
            
            Q_matrices.append(Q)
        
        return Q_matrices
    
    def _build_edge_heap(self, vertices: np.ndarray, faces: np.ndarray, 
                        topology: Dict, Q_matrices: List[np.ndarray]) -> List:
        """Строим приоритетную очередь ребер для схлопывания"""
        
        edge_heap = []
        edges = topology['edges']
        edge_to_faces = topology['edge_to_faces']
        
        for edge_id, edge in enumerate(edges):
            v1, v2 = edge
            
            # Проверяем boundary edges если нужно
            if self.preserve_boundary and len(edge_to_faces[edge]) < 2:
                continue
            
            # Вычисляем оптимальную позицию и ошибку
            optimal_pos, error = self._compute_edge_error(vertices, edge, Q_matrices)
            
            # Добавляем в heap (используем отрицательную ошибку для min-heap)
            heapq.heappush(edge_heap, (error, edge_id, edge, optimal_pos))
        
        return edge_heap
    
    def _compute_edge_error(self, vertices: np.ndarray, edge: Tuple[int, int], 
                           Q_matrices: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Вычисляем ошибку схлопывания ребра и оптимальную позицию"""
        
        v1, v2 = edge
        Q1, Q2 = Q_matrices[v1], Q_matrices[v2]
        Q = Q1 + Q2
        
        # Пытаемся найти оптимальную позицию решением системы
        try:
            Q_upper = Q[:3, :3]
            if np.abs(np.linalg.det(Q_upper)) > 1e-10:
                # Решаем систему Q_upper * v = -Q[:3, 3]
                optimal_pos = np.linalg.solve(Q_upper, -Q[:3, 3])
            else:
                # Fallback к midpoint
                optimal_pos = 0.5 * (vertices[v1] + vertices[v2])
        except:
            # Fallback к midpoint
            optimal_pos = 0.5 * (vertices[v1] + vertices[v2])
        
        # Вычисляем ошибку
        v_homogeneous = np.append(optimal_pos, 1.0)
        error = np.dot(v_homogeneous, np.dot(Q, v_homogeneous))
        
        return optimal_pos, error
    
    def _collapse_edges(self, vertices: np.ndarray, faces: np.ndarray, topology: Dict,
                       Q_matrices: List[np.ndarray], edge_heap: List, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
        """Выполняем схлопывание ребер"""
        
        current_faces = len(faces)
        
        # Маски для активных вершин и граней
        vertex_active = np.ones(len(vertices), dtype=bool)
        face_active = np.ones(len(faces), dtype=bool)
        
        # Копии для изменения
        vertices_copy = vertices.copy()
        
        # Словарь переназначения вершин
        vertex_mapping = {i: i for i in range(len(vertices))}
        
        collapsed_edges = set()
        
        while current_faces > target_faces and edge_heap:
            # Берем ребро с минимальной ошибкой
            error, edge_id, edge, optimal_pos = heapq.heappop(edge_heap)
            
            v1, v2 = edge
            
            # Проверяем что вершины еще активны
            if not vertex_active[v1] or not vertex_active[v2]:
                continue
            
            # Проверяем что ребро не было схлопнуто
            if edge in collapsed_edges:
                continue
            
            # Проверяем валидность схлопывания
            if not self._is_collapse_valid(v1, v2, topology, vertex_active, face_active):
                continue
            
            # Выполняем схлопывание
            faces_to_remove = self._collapse_edge(v1, v2, optimal_pos, vertices_copy, faces, 
                                                topology, vertex_active, face_active, vertex_mapping)
            
            current_faces -= len(faces_to_remove)
            collapsed_edges.add(edge)
            
            # Обновляем Q-матрицу для оставшейся вершины
            Q_matrices[v1] = Q_matrices[v1] + Q_matrices[v2]
            
            # Добавляем новые ребра в heap
            self._update_edge_heap(v1, vertices_copy, topology, Q_matrices, edge_heap, 
                                 vertex_active, collapsed_edges)
        
        # Собираем финальный результат
        final_vertices, final_faces = self._build_final_mesh(vertices_copy, faces, 
                                                           vertex_active, face_active, vertex_mapping)
        
        return final_vertices, final_faces
    
    def _is_collapse_valid(self, v1: int, v2: int, topology: Dict, 
                          vertex_active: np.ndarray, face_active: np.ndarray) -> bool:
        """Проверяем можно ли схлопнуть ребро"""
        
        if not self.preserve_topology:
            return True
        
        # Проверяем что у ребра есть общие соседи
        neighbors_v1 = set(n for n in topology['v2v'][v1] if vertex_active[n])
        neighbors_v2 = set(n for n in topology['v2v'][v2] if vertex_active[n])
        
        shared_neighbors = neighbors_v1.intersection(neighbors_v2)
        shared_neighbors.discard(v1)
        shared_neighbors.discard(v2)
        
        # Для manifold mesh должно быть ровно 2 общих соседа
        if len(shared_neighbors) != 2:
            return False
        
        return True
    
    def _collapse_edge(self, v1: int, v2: int, optimal_pos: np.ndarray, vertices: np.ndarray,
                      faces: np.ndarray, topology: Dict, vertex_active: np.ndarray, 
                      face_active: np.ndarray, vertex_mapping: Dict) -> List[int]:
        """Схлопываем ребро v1-v2"""
        
        # Обновляем позицию v1
        vertices[v1] = optimal_pos
        
        # Деактивируем v2
        vertex_active[v2] = False
        
        # Находим грани для удаления (те что содержат и v1 и v2)
        faces_to_remove = []
        
        shared_faces = topology['vf'][v1].intersection(topology['vf'][v2])
        
        for face_id in shared_faces:
            if face_active[face_id]:
                face_active[face_id] = False
                faces_to_remove.append(face_id)
        
        # Обновляем ссылки на v2 -> v1 в оставшихся гранях
        for face_id in topology['vf'][v2]:
            if face_active[face_id]:
                face = faces[face_id]
                for i in range(3):
                    if face[i] == v2:
                        face[i] = v1
        
        # Обновляем топологию
        topology['vf'][v1].update(topology['vf'][v2])
        topology['vf'][v1] -= set(faces_to_remove)
        topology['vf'][v2].clear()
        
        topology['v2v'][v1].update(topology['v2v'][v2])
        topology['v2v'][v1].discard(v1)
        topology['v2v'][v1].discard(v2)
        
        # Обновляем связи в соседних вершинах
        for neighbor in topology['v2v'][v2]:
            if vertex_active[neighbor]:
                topology['v2v'][neighbor].discard(v2)
                topology['v2v'][neighbor].add(v1)
        
        topology['v2v'][v2].clear()
        
        # Обновляем mapping
        vertex_mapping[v2] = v1
        
        return faces_to_remove
    
    def _update_edge_heap(self, v1: int, vertices: np.ndarray, topology: Dict,
                         Q_matrices: List[np.ndarray], edge_heap: List,
                         vertex_active: np.ndarray, collapsed_edges: Set) -> None:
        """Обновляем приоритетную очередь после схлопывания"""
        
        # Добавляем новые ребра от v1 к его соседям
        for neighbor in topology['v2v'][v1]:
            if vertex_active[neighbor] and neighbor != v1:
                edge = tuple(sorted([v1, neighbor]))
                
                if edge not in collapsed_edges:
                    optimal_pos, error = self._compute_edge_error(vertices, edge, Q_matrices)
                    heapq.heappush(edge_heap, (error, -1, edge, optimal_pos))
    
    def _build_final_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                         vertex_active: np.ndarray, face_active: np.ndarray,
                         vertex_mapping: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Строим финальный упрощенный меш"""
        
        # Собираем активные вершины
        active_vertices = vertices[vertex_active]
        
        # Создаем mapping старых индексов в новые
        vertex_remap = {}
        new_idx = 0
        for old_idx in range(len(vertices)):
            if vertex_active[old_idx]:
                vertex_remap[old_idx] = new_idx
                new_idx += 1
        
        # Собираем активные грани с новыми индексами
        active_faces = []
        for face_id, face in enumerate(faces):
            if face_active[face_id]:
                new_face = []
                valid_face = True
                
                for v_idx in face:
                    # Следуем по цепочке mapping'а
                    final_v = v_idx
                    while final_v in vertex_mapping and vertex_mapping[final_v] != final_v:
                        final_v = vertex_mapping[final_v]
                    
                    if final_v in vertex_remap:
                        new_face.append(vertex_remap[final_v])
                    else:
                        valid_face = False
                        break
                
                # Проверяем что грань не вырожденная
                if valid_face and len(set(new_face)) == 3:
                    active_faces.append(new_face)
        
        return active_vertices, np.array(active_faces, dtype=np.int32)


def advanced_reduce_faces(mesh_obj: Union[Mesh, FastMesh], target_faces: int,
                         preserve_boundary: bool = True, preserve_topology: bool = True) -> Union[Mesh, FastMesh]:
    """
    Продвинутое уменьшение граней с использованием Quadric Edge Collapse
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        target_faces: целевое количество граней
        preserve_boundary: сохранять границы меша
        preserve_topology: сохранять топологию меша
        
    Returns:
        Упрощенный mesh объект
    """
    simplifier = EdgeCollapseSimplifier(preserve_boundary, preserve_topology)
    return simplifier.simplify(mesh_obj, target_faces)


class AdvancedFaceReducer:
    """Продвинутый класс для уменьшения граней"""
    
    def __init__(self, target_faces: int = 200000, preserve_boundary: bool = True, 
                 preserve_topology: bool = True):
        self.target_faces = target_faces
        self.preserve_boundary = preserve_boundary
        self.preserve_topology = preserve_topology
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return advanced_reduce_faces(mesh_obj, self.target_faces, 
                                   self.preserve_boundary, self.preserve_topology)
