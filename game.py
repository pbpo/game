import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.SysFont(None, 24)

drawing = False
points = []
tbox_w, tbox_h = 200, 50

def fft_smooth(points, keep_fraction=0.1):
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    n = len(points)
    x_fft = np.fft.fft(x_coords)
    y_fft = np.fft.fft(y_coords)
    keep = int(n * keep_fraction / 2)
    x_fft[keep:-keep] = 0
    y_fft[keep:-keep] = 0
    x_smooth = np.fft.ifft(x_fft).real
    y_smooth = np.fft.ifft(y_fft).real
    smoothed_points = list(zip(x_smooth, y_smooth))
    return smoothed_points

def crofton_length(screen, step=10):
    
    img = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
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

def main():
    global drawing, points
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
                    screen.fill((0, 0, 0))
                    points = []
                    pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    if len(points) > 2:
                        smoothed_points = fft_smooth(points)
                        screen.fill((0, 0, 0))

            
                        pygame.draw.lines(screen, (255, 255, 255), False, points, 3)
                        pygame.display.flip()
                        crofton_curve_length = crofton_length(screen)

         
                        smoothed_surface = pygame.Surface(screen.get_size())
                        smoothed_surface.fill((0, 0, 0))
                        pygame.draw.lines(smoothed_surface, (255, 255, 255), False, smoothed_points, 3)
                        smoothed_length = crofton_length(smoothed_surface)

         
                        text_length = font.render(f'Original Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                        text_smoothed = font.render(f'Smoothed Length: {smoothed_length:.1f}', True, (255, 255, 255))

           
                        pygame.draw.rect(screen, (0, 0, 0), (0, 0, tbox_w, tbox_h))
                        screen.blit(smoothed_surface, (0, 0))

           
                        screen.blit(text_length, (10, 5))
                        screen.blit(text_smoothed, (10, 30))

           
                        

                        pygame.display.flip()
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                points.append((x, y))
                if len(points) > 1:
                    pygame.draw.line(screen, (255, 255, 255), points[-2], points[-1], 3)
                pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()