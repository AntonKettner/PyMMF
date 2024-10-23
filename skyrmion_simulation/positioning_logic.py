import numpy as np
import logging

from numpy.linalg import norm as value

def positioning(stat_tracker, index_t, sim, spin, GPU, ):
    # get the current x and y coordinates of the skyrmion
    x_0 = stat_tracker[index_t]["x0"]
    y_0 = stat_tracker[index_t]["y0"]

    # get the movement of the skyrmion in the last timestep
    try:
        x_min_1 = stat_tracker[index_t - 1]["x0"]
        y_min_1 = stat_tracker[index_t - 1]["y0"]
        movement = np.array([x_0 - x_min_1, y_0 - y_min_1])
    except (IndexError, KeyError) as e:
        logging.debug(f"Could not retrieve previous position: {e}")
        movement = np.array([0.0, 0.0])  # Default value if there is an error

    # get v_s responsible for current movement
    prev_v_s_x = stat_tracker[index_t - 1]["v_s_x"]
    prev_v_s_y = stat_tracker[index_t - 1]["v_s_y"]

    # get the error of the skyrmion position to the set position
    error = np.array([x_0 - spin.skyr_set_x, y_0 - spin.skyr_set_y])
    error_value = value(error)
    stat_tracker[index_t]["error"] = error

    # set the error limit
    local_max_error = sim.max_error * min(float(value([prev_v_s_x, prev_v_s_y])) ** 0.3 * spin.r_skyr**2, 1)

    # FIRST STEP: gather the distance that the skyrion has moved in time t by just drifting without current
    if index_t == 0:

        # drift distance
        delta_r_native = np.array([x_0 - spin.skyr_set_x, y_0 - spin.skyr_set_y])
        logging.info(f"delta_r_native: {delta_r_native}")
        stat_tracker[index_t]["v_s_x"] = 0
        stat_tracker[index_t]["v_s_y"] = 0

        # set v_s to -100 to test relation to movement
        v_s_x = -100
        v_s_y = 0
        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

        # copy v_s array to GPU
        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
        tex.set_array(GPU.cuda_v_s)
        cuda.Context.synchronize()

        # 2D v_s spacial info
        v_s_strength = value([v_s_x, v_s_y])
        v_s_angle = np.degrees(np.arctan2(v_s_y, v_s_x))
        logging.info(f"v_s_strength, v_s_angle: {v_s_strength, v_s_angle}")

        # reset the spinfield to relaxed_init_spins
        cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins.copy())
        cuda.Context.synchronize()

        # deduce one from skyr_conter to set skyr again
        skyr_counter -= 1

    # index 1: with v_s from index 0: save the amount of movement compared to the v_s_factor
    elif index_t == 1:

        # movement factor
        del_x_by_v_s = (np.array([x_0 - spin.skyr_set_x, y_0 - spin.skyr_set_y]) - delta_r_native) / v_s_x
        logging.info(f"del_x_by_v_s_10: {del_x_by_v_s}")

        # set v_s to 0 for first try
        v_s_x = 0
        v_s_y = 0

        # make array of field_size_x and field_size_y out of v_s_x and v_s_y
        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

        # 2D spacial info
        v_s_strength = value([v_s_x, v_s_y])
        v_s_angle = np.arctan2(v_s_y, v_s_x)
        logging.info(f"v_s_strength, v_s_angle: {v_s_strength, v_s_angle}")

        # copy v_s array to GPU
        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
        tex.set_array(GPU.cuda_v_s)
        cuda.Context.synchronize()

        # reset the spinfield to relaxed_init_spins
        GPU.spins_evolved = relaxed_init_spins.copy()
        cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins.copy())
        cuda.Context.synchronize()

        # deduce one from skyr_conter to set skyr again
        skyr_counter -= 1

        # set the next position
        spin.skyr_set_x = sim.distances[0]

    # ITERATION LOOP
    elif error_value > local_max_error:

        # if skyrmion is deleted at wall
        if not 0.8 < np.abs(q) < 1.2:
            logging.warning("Skyrmion is entering the wall")

            # logging.warning(f"Skyrmion last position seems wrong, replaced by the one before")
            last_skyr_spinfield = temp_last_skyr_spinfield.copy()

            # reset the learning rate cycle
            t_one_pos = 0

            # count the consecutive skyr eliminations in one position
            skyr_elims += 1

            # more then 3 eliminations at one position before reaching end
            if skyr_elims > 3:

                # potential next step size
                next_step_size = (sim.distances[index_now] - sim.distances[index_now - 1]) / 10

                if next_step_size < 0.01:
                    logging.warning(f"{spin.skyr_set_x} is the furthest that the skyrmion is not stable anymore")
                    stat_tracker[start_v_s_x_y_deletion_index:]["v_s_x"] = 0
                    stat_tracker[start_v_s_x_y_deletion_index:]["v_s_y"] = 0
                    break
                else:
                    logging.warning(f"Skyrmion is destroyed at {spin.skyr_set_x}, increasing the position density")

                    # ---- get the step distance right now ----
                    index_now = np.where(sim.distances == spin.skyr_set_x)[0][0]

                    # set new_positions to be of higher density then before
                    old_distances = sim.distances[:index_now]
                    start = old_distances[-1]
                    stop = sim.distances[-1]
                    new_distances = np.arange(start + next_step_size, stop, next_step_size)

                    # concatenate the old distances with the new distances
                    sim.distances = np.concatenate((old_distances, new_distances))

                    # set the skyr_set_x to the new position
                    spin.skyr_set_x = int(sim.distances[index_now])

                    logging.warning(f"skyr_set_x now: {spin.skyr_set_x}")
                    logging.warning(f"new distances: {new_distances}")

        # error away from dest_position
        old_error = stat_tracker[index_t - 1]["error"]

        # error is smaller than the smallest error yet and smaller than the max error * 100
        if error_value < smallest_error_yet:
            # set this as new best error --> load in as spinfield
            logging.info(f"new best error: {error_value} setting this as starting spinfield")
            temp_last_skyr_spinfield = last_skyr_spinfield.copy()
            last_skyr_spinfield = GPU.spins_evolved.copy()
            smallest_error_yet = error_value
            logging.info(f"resetting error_streak_counter and cyclic learning rate")
            # error_streak_counter = 0
            t_one_pos = 0

        # LASTLY THERE WAS A STREAK
        if error_streak_counter >= 1:
            learning_rate = np.array([0.1, 0.1])
            t_one_pos = 0
            logging.warning("Adjusting learning rate: 0.1")

        # v_s has non 0 component(s)
        elif np.any(v_s != 0):
            learning_rate = spin.calculate_learning_rate(t_one_pos)
            # learning_rate = np.array([0.8, 0.8])

        # CALCULATE NEW V_S
        v_s_x = prev_v_s_x - (error[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
        v_s_y = prev_v_s_y - (error[1]) / del_x_by_v_s[1] * learning_rate[1] * lr_adjustment    #! del_x_by_v_s[0] war vorher

        logging.info(f"at t= {t:.6g} v_s_x, v_s_y, error[0], error[1], learning_rate[0]: {v_s_x, v_s_y, error[0], error[1], learning_rate[0]}")

        # if the skyrmion has just been eliminated at the edge
        if t_one_pos == 0 and skyr_elims > 0:
            v_s_x /= 2

        # log the new v_s
        stat_tracker[index_t]["v_s_x"] = v_s_x
        stat_tracker[index_t]["v_s_y"] = v_s_y

        # make array of field_size_x and field_size_y out of v_s_x and v_s_y
        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

        # copy v_s array to GPU
        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
        tex.set_array(GPU.cuda_v_s)
        cuda.Context.synchronize()

        # if too many pictures have passed revert to the last_skyr_spinfield before this one
        if t_one_pos > sim.No_sim_img / 20:
            if reverts < 2:
                logging.warning(f"Skyrmion last position seems wrong, replaced by the one before")
                last_skyr_spinfield = temp_last_skyr_spinfield.copy()
                # adjust the learning rate more slowly
                lr_adjustment *= 0.3
                t_one_pos = 0
                reverts += 1
            else:
                logging.warning(f"Skyrmion last position seems wrong AGAIN, loop broken")
                break

        # reset the spinfield and place skyrmion at location x, y
        cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
        cuda.Context.synchronize()

        # increment t_one_pos, reset the error_streak_counter
        error_streak_counter = 0
        t_one_pos += 1

    # WHEN ERROR IS IN RANGE
    else:

        # FINAL POSITION IS NOT REACHED
        if not error_streak_counter >= sim.cons_reach_threashold:
            logging.warning(f"{error_streak_counter + 1} reaches at X={spin.skyr_set_x} with (vsx, vsy): ({v_s_x}, {v_s_y})")

            # error is smaller than the smallest error yet and smaller than the max error * 10
            if error_value < smallest_error_yet and error_value < local_max_error * 10:
                # set this as new best error --> load in as spinfield
                logging.info(f"new best error: {error_value} setting this as starting spinfield")
                temp_last_skyr_spinfield = last_skyr_spinfield.copy()
                last_skyr_spinfield = GPU.spins_evolved.copy()
                smallest_error_yet = error_value
                logging.info(f"resetting error_streak_counter and cyclic learning rate")
                # error_streak_counter = 0
                t_one_pos = 0

            # NOT 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED
            if reset:
                # big learning rate
                learning_rate = np.array([0.1, 0.1])

                # Rely on error to calculate new v_s
                v_s_x = prev_v_s_x - (error[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                v_s_y = prev_v_s_y - (error[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment

                # make array from v_s_x and v_s_y
                v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                # copy v_s array to GPU
                GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                tex.set_array(GPU.cuda_v_s)

                # reset the spinfield and place skyrmion at location x, y
                cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
                cuda.Context.synchronize()
            else:
                # small learning rate
                learning_rate = np.array([0.1, 0.1])

                # Rely on movement to calculate new v_s
                v_s_x = prev_v_s_x - (movement[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                v_s_y = prev_v_s_y - (movement[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment

                # make array of field_size_x and field_size_y out of v_s_x and v_s_y
                v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

            # track the new v_s
            stat_tracker[index_t]["v_s_x"] = v_s_x
            stat_tracker[index_t]["v_s_y"] = v_s_y

            # reset the spinfield and place skyrmion at location x, y
            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)

            # increment the consecutive_reaches
            error_streak_counter += 1
            t_one_pos += 1

        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the first time
        elif error_streak_counter >= sim.cons_reach_threashold and reset:
            logging.warning(f"POTENTIAL V_S REACHED")

            # reset the spinfield and place skyrmion at location x, y
            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
            cuda.Context.synchronize()

            # set counters
            reset = False
            error_streak_counter = 0
            t_one_pos = 0

        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the second time
        elif error_streak_counter >= sim.cons_reach_threashold and not reset:

            # angle of v_s
            theta_deg = np.degrees(np.arctan(v_s_y / v_s_x))
            logging.warning(f"Skyrmion stays at X={spin.skyr_set_x} with (vsx, vsy): ({v_s_x}, {v_s_y})")
            logging.warning(f"angle at {t:011.6f} ns: {theta_deg}")
            logging.warning(f"error at {t:011.6f} ns: {stat_tracker[index_t]['error']}")
            logging.warning(f"max_error: {local_max_error}")

            # CALCULATE FINAL V_S
            v_s_x_last_n = stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_x"]
            v_s_x_avg = np.average(v_s_x_last_n)
            v_s_y_last_n = stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_y"]
            v_s_y_avg = np.average(v_s_y_last_n)
            r_last_n = stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["r1"]
            r_avg = np.average(r_last_n)
            logging.info(f"vsx_avg: {v_s_x_avg}")
            logging.info(f"vsy_avg: {v_s_y_avg}")
            logging.info(f"last 5 vsx: {v_s_x_last_n}")
            logging.info(f"last 5 vsy: {v_s_y_last_n}")

            # track the final v_s and r
            stat_tracker[index_t]["v_s_x"] = v_s_x_avg
            stat_tracker[index_t]["v_s_y"] = v_s_y_avg
            stat_tracker[index_t]["r1"] = r_avg

            # reset the values of stat_tracker before index_t
            stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_x"] = 0
            stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_y"] = 0

            # set counters
            reset = True
            error_streak_counter = 0
            skyr_elims = 0
            lr_adjustment = 1
            t_one_pos = 0

            # NEW POSITION AVAILABLE
            if spin.skyr_set_x < sim.distances[-1]:

                # get the new position
                index_now = np.where(sim.distances == spin.skyr_set_x)[0][0]
                spin.skyr_set_x = int(sim.distances[index_now + 1].item())
                logging.warning(f"position {index_now + 1} of {len(sim.distances)} reached")
                logging.warning(f"NEW X: {spin.skyr_set_x}")

                # set v_s to 0
                spin.update_current(v_s_sample_factor=0, bottom_angle=0)

                # copy v_s array to GPU
                GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                tex.set_array(GPU.cuda_v_s)
                cuda.Context.synchronize()

                # afterwards set the deletion index to the current index + 1
                start_v_s_x_y_deletion_index = index_t + 1

                # do not change vs simply track vsx and vsy
                stat_tracker[index_t]["v_s_x"] = v_s_x
                stat_tracker[index_t]["v_s_y"] = v_s_y

                # No eliminations yet
                if skyr_elims == 0:

                    # reset the spinfield and place skyrmion at location x, y
                    skyr_counter -= 1
                    cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins)
                    cuda.Context.synchronize()

                # reset the error_counter
                smallest_error_yet = 1000

            else:
                logging.warning("Skyrmion is at the wall")
                break