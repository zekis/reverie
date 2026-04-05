"""Live state view across all experts. Read-only observer of loader."""


class Registry:
    def __init__(self, loader):
        self.loader = loader

    def warm_lemmas(self) -> list[str]:
        return list(self.loader.warm.keys())

    def active(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        return sorted(
            [(l, e.activation) for l, e in self.loader.warm.items()
             if e.activation > threshold],
            key=lambda x: -x[1],
        )

    def stats(self) -> dict:
        return self.loader.stats()
