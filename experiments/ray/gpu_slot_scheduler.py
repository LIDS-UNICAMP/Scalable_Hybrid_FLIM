# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""Escalonador de slots de GPU do launcher Ray.

O esqueleto veio verbatim de ``scripts/run_ssl_ray.py:349-382``, que so sabia
tratar ``max_per_gpu`` como int uniforme. As duas capacidades que faltavam
foram trazidas das OUTRAS copias vivas, tambem verbatim:

- slots heterogeneos por GPU (``max_per_gpu`` aceitando ``dict[int, int]``):
  ``scripts/distillation_conv_ray.py:492-536``;
- cota total por GPU (``quota`` / ``_assigned`` / ``_available``):
  ``scripts/autoencoder_flim_ray.py:515-561`` e
  ``scripts/classification_flim_ray.py:326-380`` (as duas sao a mesma
  implementacao; so a docstring difere).

Sustenta o desenho de GPU do repo: o Ray sobe com ``num_gpus=0`` e cada task
recebe um ``gpu_id`` e fixa ``CUDA_VISIBLE_DEVICES`` por conta propria. E por
isso que NAO se usa ``@ray.remote(num_gpus=1)`` - isso tiraria do usuario a
escolha de QUAL GPU e impediria mais de um job por GPU (``max_per_gpu``).
"""
from __future__ import annotations

from typing import Optional


class GpuSlotScheduler:
    """Slots de concorrencia por GPU, opcionalmente limitados por uma cota total.

    ``max_per_gpu`` limita quantos experimentos rodam AO MESMO TEMPO numa GPU.
    Aceita ``int`` - o mesmo limite para todas, o comportamento de sempre - ou
    ``dict[int, int]``, um limite por GPU (``{0: 7, 1: 3}``): e a capacidade que
    ``--gpu-slots`` dava nos lancadores antigos.

    ``quota`` (``resources.max_total_per_gpu``) limita quantos experimentos
    aquela GPU recebe NO TOTAL na fila inteira; a GPU com cota esgotada nunca
    mais e escolhida, mesmo com slot livre. Cota 0 deixa a GPU de fora. Sem
    ``quota`` a fila e drenada de forma gulosa (quem liberar primeiro pega o
    proximo experimento).
    """

    def __init__(self, gpu_ids: list[int], max_per_gpu: "int | dict[int, int]",
                 quota: Optional[dict[int, int]] = None) -> None:
        # Aceita int uniforme OU dict por GPU: {0: 7, 1: 3}
        if isinstance(max_per_gpu, int):
            self._limits: dict[int, int] = {gid: max_per_gpu for gid in gpu_ids}
        else:
            self._limits = dict(max_per_gpu)
        self._running: dict[int, int] = {gid: 0 for gid in gpu_ids}
        self._quota: Optional[dict[int, int]] = dict(quota) if quota else None
        self._assigned: dict[int, int] = {gid: 0 for gid in gpu_ids}

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._running)

    def _available(self, gid: int) -> bool:
        if self._running[gid] >= self._limits[gid]:
            return False
        if self._quota is not None and self._assigned[gid] >= self._quota.get(gid, 0):
            return False
        return True

    def pick_gpu(self) -> Optional[int]:
        candidates = [(self._running[gid], gid) for gid in self._running if self._available(gid)]
        return min(candidates)[1] if candidates else None

    def acquire(self, gpu_id: int) -> None:
        self._running[gpu_id] += 1
        self._assigned[gpu_id] += 1

    def release(self, gpu_id: int) -> None:
        self._running[gpu_id] = max(0, self._running[gpu_id] - 1)

    def total_running(self) -> int:
        return sum(self._running.values())

    def total_slots(self) -> int:
        """Concorrencia maxima da fila. Com slots heterogeneos nao ha
        `len(gpu_ids) * max_per_gpu` que valha: pergunte aqui."""
        return sum(self._limits.values())

    def has_free_slot(self) -> bool:
        return any(self._available(gid) for gid in self._running)

    def status_line(self) -> str:
        if self._quota is not None:
            return " | ".join(
                f"GPU {gid}: {self._running[gid]}/{self._limits[gid]} "
                f"({self._assigned[gid]}/{self._quota.get(gid, 0)})"
                for gid in self.gpu_ids
            )
        return " | ".join(
            f"GPU {gid}: {self._running[gid]}/{self._limits[gid]}"
            for gid in self.gpu_ids
        )
