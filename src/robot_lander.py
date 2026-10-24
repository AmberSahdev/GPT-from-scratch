"""A cosmetic skin over LunarLander's unchanged physics and observations."""

import math
from pathlib import Path

import numpy as np
from gymnasium.envs.box2d.lunar_lander import (
    LEG_DOWN,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
    LunarLander,
)

SPRITE_PATH = Path(__file__).resolve().parent.parent / "assets" / "optimus-prime.png"
SPRITE_HEIGHT = 80


class RobotLander(LunarLander):
    def render(self):
        import pygame

        if self.render_mode is None:
            return super().render()

        if not hasattr(self, "robot_sprite"):
            sprite = pygame.image.load(SPRITE_PATH)
            sprite = sprite.subsurface(sprite.get_bounding_rect(min_alpha=128))
            width = round(sprite.get_width() * SPRITE_HEIGHT / sprite.get_height())
            self.robot_sprite = pygame.transform.scale(sprite, (width, SPRITE_HEIGHT))

        # Hide only the stock drawing; Box2D bodies, joints and contacts stay intact.
        drawlist = self.drawlist
        render_mode = self.render_mode
        self.drawlist = []
        self.render_mode = "rgb_array"
        try:
            super().render()
        finally:
            self.drawlist = drawlist
            self.render_mode = render_mode

        # The stock renderer draws flags inside its body loop, so restore them here.
        for x in (self.helipad_x1, self.helipad_x2):
            x = round(x * SCALE)
            ground = round(VIEWPORT_H - self.helipad_y * SCALE)
            top = ground - 50
            pygame.draw.line(self.surf, "white", (x, ground), (x, top))
            pygame.draw.polygon(
                self.surf, (204, 204, 0), [(x, top), (x, top + 10), (x + 25, top + 5)]
            )

        # Place the boot soles at the original landing height and rotate with the body.
        center = self.lander.GetWorldPoint(
            (0, (SPRITE_HEIGHT / 2 - LEG_DOWN) / SCALE)
        )
        sprite = pygame.transform.rotate(self.robot_sprite, math.degrees(self.lander.angle))
        rect = sprite.get_rect(
            center=(round(center.x * SCALE), round(VIEWPORT_H - center.y * SCALE))
        )
        self.surf.blit(sprite, rect)

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(pygame.surfarray.array3d(self.surf), (1, 0, 2))
