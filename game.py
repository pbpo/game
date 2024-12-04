import pygame
import numpy as np
import random

# Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Crofton Length Game")
font = pygame.font.SysFont(None, 24)

# Game variables
drawing = False
points = []
tbox_w, tbox_h = 300, 100
total_players = 6
# Create a list of players, including the reference as 'Reference'
players = ["Reference"] + [f"Player {i}" for i in range(1, total_players + 1)]
random.shuffle(players)  # Shuffle the player order
current_player_index = 0  # Index in the shuffled players list
player_lengths = {player: None for player in players}
player_surfaces = {player: pygame.Surface(screen.get_size()) for player in players}
for surf in player_surfaces.values():
    surf.fill((0, 0, 0))

# FFT Smoothing Functio
def fft_smooth(points, keep_fraction=0.01):
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
def display_instructions(current_player):
    instructions = [
        f"{current_player}: Draw when you're ready.",
        "Left Click: Start/Continue Drawing",
        "Right Click: Clear Screen (only for Reference)",
    ]
    for idx, text in enumerate(instructions):
        rendered_text = font.render(text, True, (255, 255, 255))
        screen.blit(rendered_text, (10, 10 + idx * 20))

def main():
    global drawing, points, current_player_index
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
                    # Allow only the 'Reference' player to clear the screen
                    current_player = players[current_player_index]
                    if current_player == "Reference":
                        for surf in player_surfaces.values():
                            surf.fill((0, 0, 0))
                        for player in players:
                            player_lengths[player] = None
                        current_player_index = 0
                        screen.fill((0, 0, 0))
                        pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    if len(points) > 2:
                        smoothed_points = fft_smooth(points)
                        current_player = players[current_player_index]
                        player_surfaces[current_player].fill((0, 0, 0))

                        pygame.draw.lines(player_surfaces[current_player], (255, 255, 255), False, smoothed_points, 3)
                        crofton_curve_length = crofton_length(player_surfaces[current_player])

                        player_lengths[current_player] = crofton_curve_length

                        # Draw on the main screen
                        screen.blit(player_surfaces[current_player], (0, 0))

                        # Display text
                        if current_player == "Reference":
                            text = font.render(f'Reference Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                        else:
                            text = font.render(f'{current_player} Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                        screen.blit(text, (10, 10 + current_player_index * 20))
                        pygame.display.flip()

                        current_player_index += 1
                        if current_player_index >= len(players):
                            # Calculate results after all players have finished
                            reference_length = player_lengths.get("Reference", None)
                            if reference_length is not None:
                                differences = {}
                                for player in players:
                                    if player != "Reference" and player_lengths[player] is not None:
                                        differences[player] = abs(player_lengths[player] - reference_length)
                                if differences:
                                    min_diff = min(differences.values())
                                    winners = [player for player, diff in differences.items() if diff == min_diff]
                                    winner_text = ", ".join(winners)
                                    result_text = font.render(f'Winner(s): {winner_text} (Difference: {min_diff:.1f})', True, (255, 255, 0))
                                    screen.blit(result_text, (10, tbox_h + 20))
                                    pygame.display.flip()
                            else:
                                # No reference length was set
                                warning_text = font.render("Reference length was not set.", True, (255, 0, 0))
                                screen.blit(warning_text, (10, tbox_h + 20))
                                pygame.display.flip()
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                points.append((x, y))
                if len(points) > 1:
                    pygame.draw.line(screen, (255, 255, 255), points[-2], points[-1], 3)
                pygame.display.flip()

        # Update UI
        screen.fill((0, 0, 0), (0, 0, tbox_w, tbox_h))
        if current_player_index < len(players):
            current_player = players[current_player_index]
            display_instructions(current_player)
        else:
            # All players have finished, display final instructions or results
            pass  # Results are already displayed after all players finish
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()