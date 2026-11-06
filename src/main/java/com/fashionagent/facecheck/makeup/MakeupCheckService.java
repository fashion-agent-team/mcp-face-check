package com.fashionagent.facecheck.makeup;

import com.fashionagent.facecheck.FaceCheckException;
import com.fashionagent.facecheck.FaceCheckProperties;
import com.fashionagent.facecheck.image.Images;
import com.fashionagent.facecheck.makeup.MakeupResult.Check;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

/**
 * face_makeup_check: 이미 학습된 모델(앱 ML Kit)이 찾은 윤곽점 + 사진 픽셀로 대칭을 잽니다. GPT는 쓰지 않습니다
 * (팀 방향: 사진 판단은 학습한 모델·이미 학습된 모델로). 윤곽점이 없으면 다시 찍게 합니다.
 *
 * <p>앱은 얼굴을 실시간으로 비추며 구도를 안내하고, 잘 잡혀 멈춘 1초 동안 3~5장을 보냅니다(2026-11-05 결정).
 * 장마다 잰 값의 중간값으로 판단해서 순간 표정·흔들림에 덜 속습니다. 판단 기준값(face.symmetry.*)은
 * training/에서 라벨 붙인 셀카로 학습해 정합니다.
 */
@Service
public class MakeupCheckService {

    static final int MAX_FRAMES = 5;

    private final FaceCheckProperties props;
    private final MakeupAnalyzer analyzer;

    public MakeupCheckService(FaceCheckProperties props) {
        this.props = props;
        this.analyzer = new MakeupAnalyzer(props.symmetry());
    }

    public MakeupResult check(String imageBase64, FaceLandmarks landmarks) {
        return check(imageBase64, landmarks, List.of());
    }

    public MakeupResult check(String imageBase64, FaceLandmarks landmarks, List<MakeupFrame> more) {
        List<MakeupFrame> frames = new ArrayList<>();
        frames.add(new MakeupFrame(imageBase64, landmarks));
        if (more != null) {
            more.stream().filter(f -> f != null).forEach(frames::add);
        }
        if (frames.size() > MAX_FRAMES) {
            throw FaceCheckException.badRequest("사진은 한 번에 " + MAX_FRAMES + "장까지 보내 주세요.");
        }
        List<MakeupResult> results = new ArrayList<>();
        for (MakeupFrame f : frames) {
            BufferedImage img = Images.decode(f.imageBase64(), props.maxImageBytes().toBytes());
            if (f.landmarks() == null || !f.landmarks().complete()) {
                throw new FaceCheckException("no_landmarks", "얼굴을 찾지 못했어요. 폰을 얼굴 정면에 다시 들어 주세요.");
            }
            results.add(analyzer.analyze(f.landmarks(), img));
        }
        return results.size() == 1 ? results.getFirst() : combine(results, props.symmetry().maxNotes());
    }

    /**
     * 여러 장 합치기. 절반 넘게 다시 찍어야 하는 장면이면 그 안내(가장 많이 나온 것)를 돌려주고,
     * 아니면 쓸 수 있는 장면들의 항목별 중간값(왼쪽 +, 오른쪽 -)으로 다시 판단합니다.
     */
    static MakeupResult combine(List<MakeupResult> results, int maxNotes) {
        List<MakeupResult> usable = results.stream().filter(r -> r.symmetry() != null).toList();
        if (usable.size() * 2 <= results.size()) {
            return results.stream().filter(r -> r.symmetry() == null)
                    .collect(Collectors.groupingBy(r -> r.framing() + "|" + r.lighting(), LinkedHashMap::new, Collectors.toList()))
                    .values().stream().max((a, b) -> Integer.compare(a.size(), b.size())).orElseThrow().getFirst();
        }
        // 절반 넘는 장면에서 건너뛴 항목은 건너뜀으로
        Map<String, Long> skipCount = usable.stream().flatMap(r -> r.symmetry().skipped().stream())
                .collect(Collectors.groupingBy(Function.identity(), LinkedHashMap::new, Collectors.counting()));
        List<String> skipped = skipCount.entrySet().stream().filter(e -> e.getValue() * 2 > usable.size())
                .map(Map.Entry::getKey).toList();

        Map<String, List<Check>> byName = usable.stream().flatMap(r -> r.symmetry().checks().stream())
                .filter(c -> !skipped.contains(c.name()))
                .collect(Collectors.groupingBy(Check::name, LinkedHashMap::new, Collectors.toList()));
        List<Check> merged = new ArrayList<>();
        byName.forEach((name, list) -> {
            double[] signed = list.stream().mapToDouble(c -> MakeupAnalyzer.LEFT.equals(c.side()) ? c.value()
                    : MakeupAnalyzer.RIGHT.equals(c.side()) ? -c.value() : 0).sorted().toArray();
            String basis = list.stream().collect(Collectors.groupingBy(Check::basis, Collectors.counting()))
                    .entrySet().stream().max(Map.Entry.comparingByValue()).orElseThrow().getKey();
            MakeupAnalyzer.check(name, median(signed), list.getFirst().limit(), basis, merged);
        });
        MakeupResult r = MakeupAnalyzer.result(merged, skipped, maxNotes);
        return new MakeupResult(r.framing(), r.lighting(), new MakeupResult.Symmetry(r.symmetry().ok(),
                r.symmetry().notes(), merged, skipped, usable.size()), r.source());
    }

    private static double median(double[] sorted) {
        int n = sorted.length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }

}
