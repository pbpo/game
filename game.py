import pygame
import numpy as np

# Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Crofton Length Game")
font = pygame.font.SysFont(None, 24)

# Game variables
drawing = False
points = []
tbox_w, tbox_h = 300, 100
players = 6
current_player = 0  # 0 is the reference line, 1-6 are participants
player_lengths = [None] * players
player_surfaces = [pygame.Surface(screen.get_size()) for _ in range(players)]
for surf in player_surfaces:
    surf.fill((0, 0, 0))

# FFT Smoothing Function
def fft_smooth(points, keep_fraction=0.1):
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    n = len(points)
    if n == 0:
        return []
    x_fft = np.fft.fft(x_coords)
    y_fft = np.fft.fft(y_coords)
    keep = int(n * keep_fraction / 2)
    x_fft[keep:-keep] = 0
    y_fft[keep:-keep] = 0
    x_smooth = np.fft.ifft(x_fft).real
    y_smooth = np.fft.ifft(y_fft).real
    smoothed_points = list(zip(x_smooth, y_smooth))
    return smoothed_points

# Crofton Length Calculation Function
def crofton_length(surface, step=10):
    img = pygame.surfarray.array3d(surface).transpose([1, 0, 2])
    height, width, _ = img.shape
    non_zero = np.any(img != 0, axis=2)
    intersections_count = 0

    for x in range(0, width, step):
        col = non_zero[:, x]
        intersections_count += count_intersections(col)

    for y in range(0, height, step):
        row = non_zero[y, :]
        intersections_count += count_intersections(row)

    estimated_length = (np.pi / 2) * intersections_count * step
    return estimated_length

def count_intersections(line):
    return np.count_nonzero(np.diff(line.astype(int)))

# Function to Display Instructions
def display_instructions():
    instructions = [
        "Player {}: Draw when you're ready.",
        "Left Click: Start/Continue Drawing",
        "Right Click: Clear Screen",
    ]
    for idx, text in enumerate(instructions):
        # For the first instruction, insert the current player number
        if idx == 0:
            rendered_text = font.render(text.format(current_player), True, (255, 255, 255))
        else:
            rendered_text = font.render(text, True, (255, 255, 255))
        screen.blit(rendered_text, (10, 10 + idx * 20))

def main():
    global drawing, points, current_player
    clock = pygame.time.Clock()
    play = True

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    points = []
                elif event.button == 3:
                    # Clear screen and reset if it's the reference line
                    if current_player == 0:
                        for surf in player_surfaces:
                            surf.fill((0, 0, 0))
                        player_lengths[:] = [None] * players
                        current_player = 0
                        screen.fill((0, 0, 0))
                        pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    if len(points) > 2:
                        smoothed_points = fft_smooth(points)
                        player_surfaces[current_player].fill((0, 0, 0))

                        pygame.draw.lines(player_surfaces[current_player], (255, 255, 255), False, smoothed_points, 3)
                        crofton_curve_length = crofton_length(player_surfaces[current_player])

                        player_lengths[current_player] = crofton_curve_length

                        # Draw on the main screen
                        screen.blit(player_surfaces[current_player], (0, 0))

                        # Display text
                        if current_player == 0:
                            text = font.render(f'Reference Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                        else:
                            text = font.render(f'Player {current_player} Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                        screen.blit(text, (10, 10 + current_player * 20))
                        pygame.display.flip()

                        current_player += 1
                        if current_player >= players:
                            # Calculate results after all players have finished
                            target_length = player_lengths[0]
                            differences = [abs(length - target_length) for length in player_lengths[1:]]
                            min_diff = min(differences)
                            winner = differences.index(min_diff) + 1  # Player numbers start at 1

                            # Display result
                            result_text = font.render(f'Winner: Player {winner} (Difference: {min_diff:.1f})', True, (255, 255, 0))
                            screen.blit(result_text, (10, tbox_h + 20))
                            pygame.display.flip()
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                points.append((x, y))
                if len(points) > 1:
                    pygame.draw.line(screen, (255, 255, 255), points[-2], points[-1], 3)
                pygame.display.flip()

        # Update UI
        screen.fill((0, 0, 0), (0, 0, tbox_w, tbox_h))
        display_instructions()
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()