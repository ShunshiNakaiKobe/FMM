program direct_1
    implicit none
    integer, parameter :: num_particles = 2
    integer :: i, j, k, num_steps
    real, parameter :: dt = 0.01  ! 時間ステップ
    real, dimension(3, num_particles) :: position, velocity, force
    real, dimension(num_particles) :: mass, charge
    real :: distance, magnitude, effective_distance
    real, parameter :: coulomb = 0.0795774715  ! クーロン定数
    real, parameter :: cutoff_distance = 100.0  ! カットオフ距離
    character(len=50) :: filename
    integer :: unit_number
    real :: start_time, end_time, total_time, average_time

    call random_seed()

    total_time = 0.0

    ! 初期条件
    do i = 1, num_particles
        do j = 1, 3
            call random_number(position(j, i))
            position(j, i) = -5000.0 + position(j, i) * 10000.0
        end do

        velocity(:, i) = 0.0

        mass(i) = 0.00001
        charge(i) = 100.0 * ((-1.0) ** i)

    end do

    position(1, 1) = -100.0
    position(2, 1) = -100.0
    position(3, 1) = -100.0
    position(1, 2) = 100.0
    position(2, 2) = 100.0
    position(3, 2) = 100.0
    print *, charge(1)
    print *, charge(2)

    num_steps = 500

    ! シミュレーションループ
    !call cpu_time(start_time)
    do k = 1, num_steps
        force = 0.0
        write(filename, "('output_step_', I4.4, '.dat')") k
        open(unit=unit_number, file=filename, status='replace')
        do i = 1, num_particles
            write(unit_number, *) position(1, i), position(2, i), position(3, i), charge(i)
        end do
        close(unit_number)
        ! 力の計算
        call cpu_time(start_time)
        do i = 1, num_particles
            do j = 1, num_particles
                if (i /= j) then
                    distance = sqrt(sum((position(:, i) - position(:, j))**2))
                    effective_distance = max(distance, cutoff_distance)
                    magnitude = coulomb * charge(i) * charge(j) / (effective_distance**2)
                    force(:, i) = force(:, i) + magnitude * (position(:, i) - position(:, j)) / effective_distance
                endif
            end do
        end do

        call cpu_time(end_time)
        total_time = total_time + (end_time - start_time)

        ! 位置と速度の更新
        do i = 1, num_particles
            position(:, i) = position(:, i) + velocity(:, i) * dt
            velocity(:, i) = velocity(:, i) + (force(:, i) / mass(i)) * dt
        end do
    end do
    !call cpu_time(end_time)

    ! 最終的な位置の出力
    do i = 1, num_particles
        print *, 'Particle', i, 'Position:', position(:, i)
    end do

    total_time = total_time + (end_time - start_time)
    !average_time = total_time / num_steps
    print *, 'Total time per step : ', total_time

end program direct_1