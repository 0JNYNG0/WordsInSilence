"""Live sensor extent, not an inferred resting-wrist/body crop. No fitting."""

def calibrate(obs, width, height, coefficient=None):
    if isinstance(width,bool) or isinstance(height,bool) or int(width)!=width or int(height)!=height or min(width,height)<=0:
        raise ValueError('Positive integer sensor size required')
    obs.validate()
    return dict(status='CALIBRATED', schema='live_sensor_extent_v1', method='actual_camera_extent',
                reason='FULL_SENSOR_NO_ARTIFICIAL_LOWER_CUTOFF', source_width=int(width),source_height=int(height),
                box_source_pixels=[0,0,int(width),int(height)],
                inferred_body_boundary_used=False,rest_wrist_required=False,
                body_size_or_posture_calibrated=False,
                note='Sensor extent only; each feature still needs its own visible usable joints. No confidence or correctness guarantee.')

def diagnostic(reasons):
    if 'FEWER_THAN16_REAL_FRAMES_NO_PADDING' in reasons:
        return '비교할 실제 영상 프레임이 부족합니다. 촬영 시작 후 동작을 마치고 종료해 주세요.'
    if 'NO_SUPPORTED_VISIBLE_FEATURES' in reasons:
        return '화면 안에서 비교에 필요한 손·팔 정보를 확인하지 못했습니다. 동작할 때 손과 팔, 어깨가 보이는지 확인해 주세요. 준비할 때 손은 화면 밖에 있어도 됩니다.'
    return '입력의 비교 근거를 확인하지 못했습니다. 부족한 항목을 확인해 주세요.'
