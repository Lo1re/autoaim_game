if explosion_effect:
    current_time = time.time()
    if current_time - explosion_start_time < explosion_duration:
        progress = (current_time - explosion_start_time) / explosion_duration
        radius = int(50 * progress)
        draw_explosion(frame, explosion_pos[0], explosion_pos[1], radius)
    else:
        explosion_effect = False

    # Виведення повідомлення про зону тепер винесено назовні
    if use_background and zone_message:
        cv2.putText(frame, zone_message, (frame_w//2 - 300, frame_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Додаємо тривалість показу повідомлення
        if time.time() - explosion_start_time > explosion_duration:
            if time.time() - explosion_start_time < explosion_duration + 3:  # Показувати 3 секунди
                cv2.putText(frame, zone_message, (frame_w//2 - 300, frame_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                zone_message = ""  # Очищення повідомлення
