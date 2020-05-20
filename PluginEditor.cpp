#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <algorithm>
#include <opencv4/opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

static bool kShowFace = true;

struct AudioPluginAudioProcessorEditor::Impl
:   public juce::CameraDevice::Listener
,   public juce::Timer
,   private juce::Thread
{
    using WF = juce::dsp::WindowingFunction<float>;
    using AP = AudioPluginAudioProcessor;
     
    Impl(AudioPluginAudioProcessorEditor *owner)
    :   juce::Thread("Image Processing Thread")
    ,   owner_(owner)
    {
        fft_ = std::make_unique<juce::dsp::FFT>(AP::kOrder);
        fft_buffer_.resize(AP::kNumBins * 2);
        wnd_ = std::make_unique<WF>(AP::kNumBins, WF::blackman);
        
        mtx_call_async_ = std::make_shared<std::mutex>();
    }
    
    ~Impl()
    {
        assert(juce::Thread::isThreadRunning() == false);
    }
    
    void startImageProcessingThread()
    {
        juce::Thread::startThread();
    }
    
    void stopImageProcessingThread()
    {
        auto p = mtx_call_async_;
        {
            assert(p);
            // call_async 処理中はデストラクタの処理をブロックする。
            std::unique_lock lock(*p);
            
            mtx_call_async_.reset();
        }
        
        signalThreadShouldExit();
        juce::Thread::stopThread(1000);
    }
    
    void onResized()
    {
        auto b = owner_->getBounds();
        
        std::unique_lock lock(gui_mtx_);
        w = b.getWidth();
        h = b.getHeight();
    }
    
    struct FaceData
    {
        juce::Image image_;                                     //!< gui で描画する画像
        std::vector<std::array<dlib::point, 3>> mouth_points_;  //!< gui で描画する口の輪郭
        std::array<double, 3> mar_ { 0.0, 0.0, 0.0 };
        int num_skipped_ = 0;
        int num_processed_ = 0;
        
        double get_average_mar() const {
            return std::reduce(mar_.begin(), mar_.end(), 0.0) / mar_.size();
        };
    };
    
    FaceData getFaceData() const
    {
        std::unique_lock lock(gui_mtx_);
        return face_data_;
    }
    
    //! 対数振幅スペクトルを取得
    //! s には、 [0..fs/2] の範囲の最新の対数振幅スペクトルのデータが書き込まれる。
    using SpectrumType = std::array<float, AP::kNumBins/2 + 1>;
    
    SpectrumType const & calculateLogAmplitudeSpectrum()
    {
        std::fill(fft_buffer_.begin(), fft_buffer_.end(), 0.0);
        owner_->processorRef.getSampleHistory(fft_buffer_);
        
        wnd_->multiplyWithWindowingTable(fft_buffer_.data(), AP::kNumBins);
        
        fft_->performRealOnlyForwardTransform(fft_buffer_.data());
        
        int const N = AP::kNumBins;
        assert(N == fft_->getSize());
        
        //! 直流成分とナイキスト周波数成分 N で正規化する。
        fft_buffer_[0] /= N;
        fft_buffer_[AP::kNumBins] /= N;
        
        //! それ以外の交流成分は、 N で正規化後、 複素共役の分を考慮して 2 倍する。
        std::transform(fft_buffer_.data() + 1,
                       fft_buffer_.data() + (N-1),
                       fft_buffer_.data() + 1,
                       [&](float f) -> float { return f / (double)N * 2.0; });
        
        //! 対数振幅スペクトルを計算
        for(int i = 0, end = N/2+1; i < end ; ++i) {
            auto amp_spec = std::sqrt(fft_buffer_[i*2] * fft_buffer_[i*2] +
                                      fft_buffer_[i*2+1] * fft_buffer_[i*2+1]);
            
            if(amp_spec == 0) {
                spectrum_[i] = -120;
            } else {
                spectrum_[i] = std::max<float>(-120, std::log10(amp_spec) * 20);
            }
        }
        
        return spectrum_;
    }
    
public:
    AudioPluginAudioProcessorEditor *owner_ = nullptr;
    std::unique_ptr<juce::CameraDevice> cam_;
    juce::Component *viewer_ = nullptr;
    
private:
    int w = 0;
    int h = 0;
    std::mutex mutable camera_mtx_; //!< カメラスレッドと画像処理スレッドの排他制御
    std::mutex mutable gui_mtx_;    //!< GUI スレッドと画像処理スレッドの排他制御
    juce::Image tmp_image_; //!< camera から取得した画像を、画像処理スレッドで処理するために保持しておく変数
    static constexpr int kMouthPointsBegin = 48;
    static constexpr int kMouthPointsEnd = 68;
        
    std::vector<float> fft_buffer_;
    std::unique_ptr<juce::dsp::FFT> fft_;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> wnd_;
    
    FaceData face_data_;
    SpectrumType spectrum_;
    
    // call_async によるメインスレッドでの非同期処理を呼び出している間はこのミューテックスをロックする。
    std::shared_ptr<std::mutex> mtx_call_async_;

private:
    void setFaceData(FaceData const &fd)
    {
        std::unique_lock lock(gui_mtx_);
        face_data_ = fd;
    }
    
    double getMouthAspectRatio(std::vector<dlib::point> const &points)
    {
        auto distance = [](dlib::point pt1, dlib::point pt2) {
            auto const x = pt2.x() - pt1.x();
            auto const y = pt2.y() - pt1.y();
            return std::sqrt(x * x + y * y);
        };
        
        auto const A = distance(points[2], points[10]);
        auto const B = distance(points[4], points[8]);
        auto const C = distance(points[0], points[6]);
        
        auto const mar = (A + B) / (2.0 * C);
        
        return mar;
    }
    
    void run() override
    {
        auto detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor predictor;

        auto tmp_file = juce::File::createTempFile("shape_predictor");
        auto os = tmp_file.createOutputStream();
        os->write(BinaryData::shape_predictor_68_face_landmarks_dat,
                  BinaryData::shape_predictor_68_face_landmarks_datSize);

        dlib::deserialize(tmp_file.getFullPathName().toStdString()) >> predictor;
        
        std::vector<dlib::point> tmp_mouth_points(kMouthPointsEnd - kMouthPointsBegin);
        cv::Mat tmp_mat;
        int tmp_w = 0;
        int tmp_h = 0;
        
        for( ; threadShouldExit() == false; ) {
            
            juce::Image tmp_image;
                       
            {
                std::unique_lock lock(camera_mtx_);
                tmp_image = std::move(tmp_image_);
            }
            
            if(tmp_image.isValid() == false) {
                sleep(10);
                continue;
            }
            
            {
                std::unique_lock resize_lock(gui_mtx_);
                if(tmp_w != w || tmp_h != h) {
                    tmp_w = w;
                    tmp_h = h;
                }
            }
            
            double image_aspect_ratio = (double)tmp_w / tmp_h;
            
            auto const actual_w = tmp_image.getWidth();
            auto const expected_w = (int)std::round(tmp_image.getHeight() * image_aspect_ratio);
            
            if(actual_w == expected_w) {
                // do nothing.
            } else if(actual_w > expected_w) {
                // 画面が横長なので左右を捨てる。
                int const nx = actual_w - expected_w;
                auto rect = tmp_image.getBounds().reduced(nx / 2.0, 0);
                tmp_image = tmp_image.getClippedImage(rect);
            } else {
                // 画面が縦長なので上下を捨てる。
                 int const ny = tmp_image.getHeight() - (tmp_image.getWidth() / image_aspect_ratio);
                 auto rect = tmp_image.getBounds().reduced(0, ny / 2.0);
                 tmp_image = tmp_image.getClippedImage(rect);
            }
             
            auto rescaled = tmp_image.rescaled(tmp_w, tmp_h);
            
//            juce::PNGImageFormat fmt;
//            juce::File file("/Users/yuasa/Desktop/juce_image.png");
//            auto fs = file.createOutputStream();
//            fmt.writeImageToStream(rescaled, *fs);
//            fs.reset();
            
            juce::Image::BitmapData bmp_rescaled(rescaled, juce::Image::BitmapData::readWrite);
            
            cv::Mat mat_rescaled;
            
            if(rescaled.getPixelData()->pixelFormat == juce::Image::PixelFormat::RGB) {
                mat_rescaled = cv::Mat(tmp_h, tmp_w, CV_8UC3, bmp_rescaled.data);
            } else if(rescaled.getPixelData()->pixelFormat == juce::Image::PixelFormat::ARGB) {
                mat_rescaled = cv::Mat(tmp_h, tmp_w, CV_8UC4, bmp_rescaled.data);
            } else {
                return;
            }
            
//            cv::imwrite("/Users/yuasa/Desktop/cv_rescaled.png", mat_rescaled);

            cv::cvtColor(mat_rescaled,
                         tmp_mat,
                         cv::COLOR_RGBA2GRAY
                         );
                        
//            auto cv_img = dlib::cv_image<dlib::rgb_pixel>(tmp_mat);
//            dlib::save_png(cv_img, "/Users/yuasa/Desktop/cv_image.png");
            
//            dlib::array2d<dlib::rgb_pixel> img;
//            dlib::assign_image(img, cv_img);
//            dlib::pyramid_up(img);
            
//            dlib::save_png(img, "/Users/yuasa/Desktop/dlib_image.png");
            
            auto img = dlib::cv_image<unsigned char>(tmp_mat);
            
            auto rects = detector(img, 0);
            
            FaceData tmp_face_data = getFaceData();
            
            if(kShowFace) {
                cv::cvtColor(tmp_mat,
                             mat_rescaled,
                             cv::COLOR_GRAY2RGBA
                             );
            }
            
            if(rects.empty()) {
                // 0 扱い。スムーズに値を減少させる。
                //tmp_face_data.image_ = std::move(rescaled);
                tmp_face_data.num_skipped_ += 1;
                setFaceData(tmp_face_data);
            } else {
                dlib::full_object_detection shape = predictor(img, rects[0]);

                for(int i = 0, end = kMouthPointsEnd - kMouthPointsBegin; i < end; ++i) {
                    tmp_mouth_points[i] = shape.part(i + kMouthPointsBegin);
                }
                
                double const tmp_mar = getMouthAspectRatio(tmp_mouth_points);
                
                if(tmp_face_data.mouth_points_.empty()) {
                    tmp_face_data.mouth_points_.resize(kMouthPointsEnd - kMouthPointsBegin);
                }
                
                for(int i = 0, end = tmp_mouth_points.size(); i < end; ++i) {
                    tmp_face_data.mouth_points_[i][0] = tmp_face_data.mouth_points_[i][1];
                    tmp_face_data.mouth_points_[i][1] = tmp_face_data.mouth_points_[i][2];
                    tmp_face_data.mouth_points_[i][2] = tmp_mouth_points[i];
                }

                tmp_face_data.mar_[0] = tmp_face_data.mar_[1];
                tmp_face_data.mar_[1] = tmp_face_data.mar_[2];
                tmp_face_data.mar_[2] = tmp_mar;
                
                tmp_face_data.num_processed_ += 1;
                
                if(kShowFace) {
                    tmp_face_data.image_ = std::move(rescaled);
                }
                
                setFaceData(tmp_face_data);
            }
            
            auto mm = juce::MessageManager::getInstance();
            mm->callAsync([this, w = std::weak_ptr<std::mutex>(mtx_call_async_)] {
                
                for( ; ; ) {
                    auto still_alive = w.lock();
                    if(!still_alive) {
                        //! デストラクタが呼び出されたあとなので、なにもせずにリターンする。
                        break;
                    }
                    
                    std::unique_lock lock(*still_alive, std::try_to_lock);
                    if(lock) {
                        owner_->OnUpdateMouthAspectRatio();
                        break;
                    }
                    
                    std::this_thread::yield();
                }
            });
        }
    }

    void imageReceived(juce::Image const &image) override
    {
        std::unique_lock lock(camera_mtx_);
        tmp_image_ = image;
    }
    
    void timerCallback() override
    {
        
    }
};

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
:   AudioProcessorEditor(&p)
,   processorRef(p)
,   pimpl_(std::make_unique<Impl>(this))
{
    juce::ignoreUnused (processorRef);
    
    int const minWidth = 120;
    int const minHeight = 90;
    int const maxWidth = 1200;
    int const maxHeight = 900;
    int const defaultWidth = 600;
    int const defaultHeight = 450;
    
    setResizable(true, true);
    setResizeLimits(minWidth, minHeight, maxWidth, maxHeight);
    
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (defaultWidth, defaultHeight);
    
    pimpl_->cam_.reset(juce::CameraDevice::openDevice(0, minWidth, minHeight, maxWidth, maxHeight));
    pimpl_->cam_->addListener(pimpl_.get());
    
    pimpl_->viewer_ = pimpl_->cam_->createViewerComponent();
    //pimpl_->viewer_->setBounds(10, 10, minWidth, minHeight);
    //addAndMakeVisible(pimpl_->viewer_);
    
    getConstrainer()->setFixedAspectRatio((double)maxWidth / maxHeight);
    setBounds(0, 0, defaultWidth, defaultHeight);
    resized();
    
    pimpl_->startImageProcessingThread();
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    pimpl_->stopImageProcessingThread();
    pimpl_->cam_->removeListener(pimpl_.get());
    delete pimpl_->viewer_;
    pimpl_.reset();
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    auto const face_data = pimpl_->getFaceData();
    
    if(kShowFace && face_data.image_.isValid()) {
        double scale = face_data.image_.getWidth() / 120.0;
        juce::AffineTransform t;
        t = t.scaled(-1, 1.0);
        t = t.scaled(1.0 / scale);
        t = t.translated(120 + 10, 10);
        g.drawImageTransformed(face_data.image_, t);
     }
    
    g.setColour (juce::Colours::white);
    g.setFont (15.0f);
    
    char const nl = '\n';
    std::stringstream ss;
    
    auto &points = face_data.mouth_points_;
    
    g.setColour(juce::Colour(0.3f, 0.7f, 0.9f, 0.75f));
    
    if(points.size() > 0) {
        int left = 10000, top = 10000, right = 0, bottom = 0;
        
        auto avg = [](auto const &pts) -> dlib::point {
            assert(pts.size() > 0);
            double sum_x = 0;
            double sum_y = 0;
            
            for(auto const &pt: pts) {
                sum_x += pt.x();
                sum_y += pt.y();
            }
            
            return {
                (int)std::round(sum_x / pts.size()),
                (int)std::round(sum_y / pts.size())
            };
        };
        
        for(auto const &pts: points) {
            left = std::min<int>(left, avg(pts).x());
            right = std::max<int>(right, avg(pts).x());
            top = std::min<int>(top, avg(pts).y());
            bottom = std::max<int>(bottom, avg(pts).y());
        }
        
        juce::Rectangle<int> b_mouth { left, top, (right - left), (bottom - top) };
        juce::Rectangle<int> b_area = getBounds();
        
        b_area = b_area.reduced((int)(b_area.getWidth() * 0.2), (int)(b_area.getHeight() * 0.2));
        
        auto const shift_mouth = b_mouth.getCentre();
        auto const shift_area = b_area.getCentre();
        
        double const scale_x = b_area.getWidth() / (double)b_mouth.getWidth();
        double const scale_y = b_area.getHeight() / (double)b_mouth.getHeight();
        double const scale = std::min<double>(scale_x, scale_y);
        
        b_mouth = b_mouth.expanded(b_mouth.getWidth() * (scale - 1.0) / 2.0,
                                   b_mouth.getHeight() * (scale - 1.0) / 2.0);
        
        auto drawLine = [&](int a, int b) {
            auto ax = avg(points[a]).x();
            auto ay = avg(points[a]).y();
            auto bx = avg(points[b]).x();
            auto by = avg(points[b]).y();
            
            ax = (ax - b_mouth.getCentre().x) * -1 + b_mouth.getCentre().x;
            bx = (bx - b_mouth.getCentre().x) * -1 + b_mouth.getCentre().x;
            
            g.drawLine((ax - shift_mouth.x) * scale + shift_area.x,
                       (ay - shift_mouth.y) * scale + shift_area.y,
                       (bx - shift_mouth.x) * scale + shift_area.x,
                       (by - shift_mouth.y) * scale + shift_area.y
                       );
        };
        
        for(int i = 0; i < 12; ++i) {
            drawLine(i, (i+1)%12);
        }
        
        for(int i = 0; i < 8; ++i) {
            drawLine(i+12, ((i+1)%8)+12);
        }
    }
    
    auto const &spectrum = pimpl_->calculateLogAmplitudeSpectrum();
    
    auto b = getBounds().reduced(30);
    for(int i = 0, end = spectrum.size()-1; i < end; ++i) {
        auto &s1 = spectrum[i];
        auto &s2 = spectrum[i+1];
        
        float x1 = log10((i / (double)end) * 200 + 1) / log10(201.0) * b.getWidth() + b.getX();
        float x2 = log10(((i+1) / (double)end) * 200 + 1) / log10(201.0) * b.getWidth() + b.getX();
        float y1 = (-s1 / (double)120) * b.getHeight() + b.getY();
        float y2 = (-s2 / (double)120) * b.getHeight() + b.getY();
        
        g.drawLine(x1, y1, x2, y2, 2.0);
    }
    
    ss
    << "Mouth Aspect Ratio : " << face_data.get_average_mar();
    
    g.drawFittedText (ss.str(), getLocalBounds().removeFromTop(100), juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    pimpl_->onResized();
}

void AudioPluginAudioProcessorEditor::OnUpdateMouthAspectRatio()
{
    auto face_data = pimpl_->getFaceData();
    
    double const kMarMin = 0.4;
    double const kMarMax = 0.95;
    
    double const mar = std::clamp(face_data.get_average_mar(), kMarMin, kMarMax);
    double const freq = (mar - kMarMin) / (kMarMax - kMarMin);
    
    auto *param = processor.getParameters()[AudioPluginAudioProcessor::Frequency];
    param->setValueNotifyingHost(freq);

    repaint();
}
