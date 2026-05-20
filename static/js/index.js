window.HELP_IMPROVE_VIDEOJS = false;

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) return;
  if (window.pageYOffset > 300) {
    scrollButton.classList.add('visible');
  } else {
    scrollButton.classList.remove('visible');
  }
});

function setupVideoCarouselAutoplay() {
  const videos = document.querySelectorAll('video');
  if (videos.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const video = entry.target;
      if (entry.isIntersecting && video.autoplay) {
        video.play().catch(() => {});
      } else if (!entry.isIntersecting && video.autoplay) {
        video.pause();
      }
    });
  }, { threshold: 0.5 });

  videos.forEach(video => observer.observe(video));
}

$(document).ready(function() {
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: true,
      autoplaySpeed: 5000,
    });
  }
  if (typeof bulmaSlider !== 'undefined') {
    bulmaSlider.attach();
  }
  setupVideoCarouselAutoplay();
});

const translations = {
  en: {
    page_title: "Research on Diffusion Policy Distillation and Acceleration for Robotic Manipulation",
    institution: "Dalian University of Technology<br>Bachelor's Thesis Project, 2026",
    button_summary: "Thesis Summary",
    button_code: "Code",
    button_videos: "Videos",
    teaser_text: "A teacher-student distillation framework for accelerating Diffusion Policy in robotic manipulation tasks.",
    abstract_title: "Abstract",
    abstract_text: "Diffusion Policy can model complex action distributions in robotic manipulation, but its iterative denoising process introduces high inference latency. This project constructs a teacher-student distillation framework to accelerate Diffusion Policy while analyzing how much teacher behavior can be preserved. A pretrained Diffusion Policy is used as the teacher, and multiple student policies are compared, including MLP-BC, GRU-BC, Transformer-BC, and Flow Matching students. Experiments are conducted on Push-T, Lift-PH, and Square-PH. The results show that student policies reduce the average action generation latency from 682.25 ms to 2.39 ms, achieving approximately 285× speedup. However, behavior preservation remains task-dependent: GRU-BC performs best on Lift-PH, Flow Matching performs best on Push-T, and Transformer-BC performs best on Square-PH.",
    method_title: "Method Overview",
    method_text: "The project follows an offline teacher-student distillation pipeline. A pretrained Diffusion Policy teacher receives observation sequences and generates future action chunks. These teacher-generated actions are then recorded as distillation targets and used to train lightweight student policies. The students are evaluated through rollout performance and average predict_action latency, allowing a direct comparison between task performance and real-time efficiency.",
    method_caption: "Overall pipeline: Diffusion Policy teacher → teacher-sample dataset → lightweight student policies → rollout evaluation and latency measurement.",
    architectures_title: "Student Policy Architectures",
    architectures_text: "Four student policy families were implemented and compared. MLP-BC provides a simple and fast deterministic baseline, GRU-BC introduces recurrent temporal modeling, Transformer-BC uses attention-based sequence modeling, and Flow Matching explores few-step generative action prediction.",
    mlp_caption: "A lightweight behavior-cloning baseline that flattens observation history and predicts an action chunk with fully connected layers.",
    gru_caption: "A recurrent student policy designed to capture temporal dependencies in observation sequences.",
    transformer_caption: "An attention-based student policy for sequence modeling and action chunk prediction.",
    fm_caption: "A few-step generative student that learns a velocity field from a source action prior to the teacher action distribution.",
    tasks_title: "Experimental Tasks",
    tasks_text: "Experiments were conducted on three robotic manipulation tasks with different control characteristics: Push-T for low-dimensional planar pushing, Lift-PH for grasping and lifting, and Square-PH for more precise manipulation and alignment.",
    pusht_desc: "Low-dimensional planar pushing task.",
    lift_desc: "Basic grasping and lifting task.",
    square_desc: "Precise manipulation, alignment, and placement task.",
    results_title: "Main Results",
    results_intro: "Student policies significantly reduced action generation latency while showing task-dependent behavior preservation. The main result is that inference speed was improved by approximately 285×, but preserving the teacher policy's behavior remained the key bottleneck.",
    metric_teacher: "Teacher latency",
    metric_student: "Average student latency",
    metric_speedup: "Overall speedup",
    table_task: "Task",
    table_best_student: "Best Student",
    table_score: "Score",
    table_retention: "Retained Teacher Score",
    table_speedup: "Speedup",
    table_caption: "Retained teacher score is computed as the best student score divided by the teacher score on the same task. Speedup is computed as teacher latency divided by student latency.",
    latency_caption: "Inference latency comparison. A logarithmic scale is recommended because the teacher is hundreds of milliseconds while students are only a few milliseconds.",
    score_caption: "Retained teacher score comparison across tasks and student policy families.",
    findings_title: "Model Strengths and Findings",
    method_col: "Method",
    strength_col: "Strength",
    limitation_col: "Limitation",
    best_case_col: "Best suited case",
    mlp_strength: "Fast and simple deterministic baseline.",
    mlp_limit: "Limited temporal modeling ability.",
    mlp_case: "Latency baseline.",
    gru_strength: "Stable recurrent temporal modeling.",
    gru_limit: "Less expressive than generative students.",
    transformer_strength: "Attention-based sequence modeling.",
    transformer_limit: "Slightly higher latency than simpler students.",
    fm_strength: "Few-step generative action prediction.",
    fm_limit: "Unstable in precise manipulation under the current design.",
    findings_text: "Overall, the experiments suggest that different student architectures are suitable for different tasks. Few-step generative students showed potential in low-dimensional planar control, while GRU-BC and Transformer-BC were more stable in manipulation tasks requiring temporal modeling and precise control.",
    rollout_title: "Teacher vs. Student Rollout Comparison",
    rollout_intro: "The following videos compare the pretrained Diffusion Policy teacher with the best-performing student policy on each task. They illustrate both the effectiveness and the remaining limitations of policy distillation.",
    teacher_label: "Teacher",
    best_student_label: "Best Student",
    pusht_task_title: "Push-T",
    lift_task_title: "Lift-PH",
    square_task_title: "Square-PH",
    pusht_task_tag: "Planar pushing",
    lift_task_tag: "Grasping and lifting",
    square_task_tag: "Precise manipulation",    
    pusht_video_caption: "Best student: <strong>FM BC-Prior(MLP) 1-step</strong>.",
    lift_video_caption: "Best student: <strong>GRU-BC</strong>.",
    square_video_caption: "Best student: <strong>Transformer-BC</strong>.",
    discussion_title: "Discussion and Limitations",
    discussion_text: "The results indicate that policy distillation can effectively reduce the inference latency of Diffusion Policy, but preserving the teacher policy's behavior remains challenging. The current experiments were mainly conducted with limited rollout repetitions, and some conclusions should therefore be interpreted as descriptive trends rather than strict statistical claims. Future work may improve prior design for Flow Matching, strengthen the expressive capacity of student models, evaluate more acceleration methods, and extend the framework to real robotic systems.",
    materials_title: "Project Materials",
    materials_text: "Supplementary materials, including implementation details, experimental tables, and rollout videos, will be organized in the project repository and linked from this page.",
    footer_project: "This page presents my bachelor's thesis project at Dalian University of Technology."
  },
  ja: {
    page_title: "ロボット操作タスクにおけるDiffusion Policyの蒸留と高速化に関する研究",
    institution: "大連理工大学<br>卒業論文プロジェクト 2026",
    button_summary: "卒業論文概要",
    button_code: "コード",
    button_videos: "動画",
    teaser_text: "ロボット操作タスクにおけるDiffusion Policyを高速化するためのteacher-student蒸留フレームワーク。",
    abstract_title: "概要",
    abstract_text: "Diffusion Policyはロボット操作における複雑な行動分布を表現できる一方で，反復的なノイズ除去過程により高い推論遅延が生じる。本プロジェクトでは，Diffusion Policyを高速化するためのteacher-student蒸留フレームワークを構築し，教師ポリシーの行動をどの程度保持できるかを分析した。事前学習済みDiffusion Policyを教師モデルとして用い，MLP-BC，GRU-BC，Transformer-BC，およびFlow Matchingに基づく学生ポリシーを比較した。実験はPush-T，Lift-PH，Square-PHの3種類のロボット操作タスクで行った。その結果，学生ポリシーは平均動作生成時間を682.25 msから2.39 msへ削減し，約285倍の高速化を達成した。一方で，教師ポリシーの行動保持性能はタスクに依存し，Lift-PHではGRU-BC，Push-TではFlow Matching，Square-PHではTransformer-BCが最良の結果を示した。",
    method_title: "手法概要",
    method_text: "本プロジェクトでは，オフラインのteacher-student蒸留パイプラインを構築した。事前学習済みDiffusion Policy教師モデルは観測系列を入力として未来のaction chunkを生成する。生成された教師行動を蒸留ターゲットとして記録し，軽量な学生ポリシーの学習に用いた。各学生モデルはrollout性能と平均predict_action遅延によって評価し，タスク性能とリアルタイム性の関係を比較した。",
    method_caption: "全体パイプライン：Diffusion Policy教師モデル → teacher-sample dataset → 軽量学生ポリシー → rollout評価と推論時間計測。",
    architectures_title: "学生ポリシーのモデル構成",
    architectures_text: "本研究では4種類の学生ポリシーを実装し比較した。MLP-BCは単純かつ高速な決定論的ベースライン，GRU-BCは再帰的な時系列モデリング，Transformer-BCは注意機構に基づく系列モデリング，Flow Matchingは少ステップの生成的行動予測を目的とする。",
    mlp_caption: "観測履歴を平坦化し，全結合層によってaction chunkを予測する軽量な行動クローニングベースライン。",
    gru_caption: "観測系列の時間依存性を捉えるための再帰型学生ポリシー。",
    transformer_caption: "系列モデリングとaction chunk予測のための注意機構ベースの学生ポリシー。",
    fm_caption: "source action priorから教師行動分布へのvelocity fieldを学習する少ステップ生成型学生モデル。",
    tasks_title: "実験タスク",
    tasks_text: "実験は制御特性の異なる3種類のロボット操作タスクで行った。Push-Tは低次元の平面押し操作，Lift-PHは把持と持ち上げ操作，Square-PHはより精密な操作・位置合わせを含むタスクである。",
    pusht_desc: "低次元の平面押し操作タスク。",
    lift_desc: "基本的な把持・持ち上げタスク。",
    square_desc: "精密な操作，位置合わせ，配置を含むタスク。",
    results_title: "主な実験結果",
    results_intro: "学生ポリシーは動作生成遅延を大幅に削減した一方で，教師ポリシーの行動保持性能にはタスク依存性が見られた。主な結果として，推論速度は約285倍改善されたが，教師行動の保持が依然として主要な課題である。",
    metric_teacher: "教師モデルの遅延",
    metric_student: "学生モデルの平均遅延",
    metric_speedup: "全体の高速化率",
    table_task: "タスク",
    table_best_student: "最良の学生モデル",
    table_score: "スコア",
    table_retention: "教師スコア保持率",
    table_speedup: "高速化率",
    table_caption: "教師スコア保持率は，同一タスクにおける最良学生スコアを教師スコアで割った値である。高速化率は，教師モデルの推論時間を学生モデルの推論時間で割った値である。",
    latency_caption: "推論遅延の比較。教師モデルは数百ミリ秒，学生モデルは数ミリ秒であるため，対数スケールでの表示が適している。",
    score_caption: "各タスクおよび学生ポリシー群における教師スコア保持率の比較。",
    findings_title: "モデルの特徴と考察",
    method_col: "手法",
    strength_col: "長所",
    limitation_col: "課題",
    best_case_col: "適したケース",
    mlp_strength: "高速で単純な決定論的ベースライン。",
    mlp_limit: "時系列モデリング能力が限定的。",
    mlp_case: "遅延比較のベースライン。",
    gru_strength: "安定した再帰的時系列モデリング。",
    gru_limit: "生成型学生モデルほどの表現力はない。",
    transformer_strength: "注意機構に基づく系列モデリング。",
    transformer_limit: "単純な学生モデルより推論遅延がやや大きい。",
    fm_strength: "少ステップの生成的行動予測。",
    fm_limit: "現在の設計では精密操作において不安定。",
    findings_text: "全体として，異なる学生モデルは異なるタスクに適していることが示された。少ステップ生成型学生モデルは低次元平面制御で可能性を示した一方，GRU-BCとTransformer-BCは時系列モデリングと精密制御を必要とする操作タスクでより安定していた。",
    rollout_title: "教師モデルと学生モデルのRollout比較",
    rollout_intro: "以下の動画では，各タスクにおける事前学習済みDiffusion Policy教師モデルと，最良の学生ポリシーを比較する。これにより，ポリシー蒸留の有効性と残された課題の両方を定性的に示す。",
    teacher_label: "教師モデル",
    best_student_label: "最良の学生モデル",
    pusht_task_title: "Push-T",
    lift_task_title: "Lift-PH",
    square_task_title: "Square-PH",
    pusht_task_tag: "平面押し操作",
    lift_task_tag: "把持・持ち上げ",
    square_task_tag: "精密操作",
    pusht_video_caption: "最良の学生モデル：<strong>FM BC-Prior(MLP) 1-step</strong>。",
    lift_video_caption: "最良の学生モデル：<strong>GRU-BC</strong>。",
    square_video_caption: "最良の学生モデル：<strong>Transformer-BC</strong>。",
    discussion_title: "考察と限界",
    discussion_text: "実験結果から，ポリシー蒸留はDiffusion Policyの推論遅延を効果的に削減できる一方で，教師ポリシーの行動保持は依然として課題であることが分かった。現在の実験は主に限られた回数のrollout評価に基づいているため，一部の結論は厳密な統計的主張ではなく記述的傾向として解釈すべきである。今後はFlow Matchingのprior設計の改善，学生モデルの表現能力向上，他の高速化手法の評価，および実ロボット環境への拡張が考えられる。",
    materials_title: "プロジェクト資料",
    materials_text: "実装詳細，実験表，rollout動画などの補足資料は，プロジェクトリポジトリに整理し，本ページからリンクする予定である。",
    footer_project: "本ページは，大連理工大学における卒業論文プロジェクトを紹介するものである。"
  }
};

function setLanguage(lang) {
  const dict = translations[lang] || translations.en;
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) {
      el.innerHTML = dict[key];
    }
  });
  document.documentElement.lang = lang === 'ja' ? 'ja' : 'en';
  const enBtn = document.getElementById('lang-en');
  const jaBtn = document.getElementById('lang-ja');
  if (enBtn && jaBtn) {
    enBtn.classList.toggle('active', lang === 'en');
    jaBtn.classList.toggle('active', lang === 'ja');
  }
  localStorage.setItem('preferredLanguage', lang);
}

document.addEventListener('DOMContentLoaded', function () {
  const savedLang = localStorage.getItem('preferredLanguage') || 'en';
  setLanguage(savedLang);
});

function setupImageViewer() {
  const viewer = document.getElementById('image-viewer');
  const viewerImg = document.getElementById('image-viewer-img');
  const closeBtn = document.querySelector('.image-viewer-close');
  const zoomInBtn = document.getElementById('zoom-in');
  const zoomOutBtn = document.getElementById('zoom-out');
  const zoomResetBtn = document.getElementById('zoom-reset');

  if (!viewer || !viewerImg) return;

  let zoom = 1;
  let baseWidth = 1000;
  let translateX = 0;
  let translateY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;

  function applyView() {
    viewerImg.style.width = `${baseWidth * zoom}px`;
    viewerImg.style.height = 'auto';
    viewerImg.style.transform = `translate(${translateX}px, ${translateY}px)`;
  }

  function resetView() {
    zoom = 1;
    translateX = 0;
    translateY = 0;
    applyView();
  }

  function openViewer(src, alt) {
    viewerImg.onload = function () {
      const viewportWidth = window.innerWidth * 0.86;
      const viewportHeight = window.innerHeight * 0.78;

      const naturalRatio =
        viewerImg.naturalWidth && viewerImg.naturalHeight
          ? viewerImg.naturalWidth / viewerImg.naturalHeight
          : 16 / 9;

      baseWidth = Math.min(viewportWidth, viewportHeight * naturalRatio);
      resetView();
    };

    viewerImg.src = src;
    viewerImg.alt = alt || '';

    viewer.classList.add('is-open');
    viewer.setAttribute('aria-hidden', 'false');
    document.body.classList.add('image-viewer-open');
  }

  function closeViewer() {
    viewer.classList.remove('is-open');
    viewer.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('image-viewer-open');
    viewerImg.src = '';
  }

  document.querySelectorAll('.zoomable-image').forEach((img) => {
    img.addEventListener('click', () => {
      openViewer(img.dataset.full || img.src, img.alt);
    });
  });

  closeBtn?.addEventListener('click', closeViewer);

  viewer.addEventListener('click', (event) => {
    if (event.target === viewer) {
      closeViewer();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && viewer.classList.contains('is-open')) {
      closeViewer();
    }
  });

  viewer.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.18 : 0.85;
      zoom = Math.min(Math.max(zoom * factor, 0.35), 10);
      applyView();
    },
    { passive: false }
  );

  viewerImg.addEventListener('mousedown', (event) => {
    isDragging = true;
    viewerImg.classList.add('is-dragging');
    startX = event.clientX - translateX;
    startY = event.clientY - translateY;
  });

  window.addEventListener('mousemove', (event) => {
    if (!isDragging) return;
    translateX = event.clientX - startX;
    translateY = event.clientY - startY;
    applyView();
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
    viewerImg.classList.remove('is-dragging');
  });

  zoomInBtn?.addEventListener('click', () => {
    zoom = Math.min(zoom * 1.25, 10);
    applyView();
  });

  zoomOutBtn?.addEventListener('click', () => {
    zoom = Math.max(zoom / 1.25, 0.35);
    applyView();
  });

  zoomResetBtn?.addEventListener('click', resetView);
}

document.addEventListener('DOMContentLoaded', function () {
  setupImageViewer();
});