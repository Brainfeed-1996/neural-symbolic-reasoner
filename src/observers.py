from typing import List, Protocol
from .rule_engine import Rule

class Observer(Protocol):
    def update(self, result: dict[str, float]):
        ...

class AuditObserver:
    def update(self, result: dict[str, float]):
        print(f"[AUDIT] Inference completed. Number of beliefs: {len(result)}")

class KnowledgeGraphObserver:
    def update(self, result: dict[str, float]):
        # Simulate updating a KG
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def notify(self, result: dict[str, float]):
        for observer in self._observers:
            observer.update(result)
