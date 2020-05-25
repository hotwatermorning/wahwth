#include "AboutDialog.h"

struct AboutDialogComponent
:   public juce::Component
{
    AboutDialogComponent()
    {
        addAndMakeVisible(lbl_plugin_name_);
        addAndMakeVisible(lbl_contact_info_);
        addAndMakeVisible(lbl_vst_copyright1_);
        addAndMakeVisible(lbl_vst_copyright2_);
        
        auto const dont_send = juce::NotificationType::dontSendNotification;
        
        lbl_plugin_name_.setText(juce::String(ProjectInfo::projectName) + " version: " + ProjectInfo::versionString, dont_send);
        lbl_contact_info_.setText("twitter: @hotwatermorning", dont_send);
        lbl_vst_copyright1_.setText("VST is a trademark of Steinberg Media Techonologies GmbH.", dont_send);
        
        setSize(400, 100);
    }
    
    void resized() override
    {
        auto b = getBounds().reduced(10);
        lbl_plugin_name_.setBounds(b.removeFromTop(20));
        b.removeFromTop(3);
        lbl_contact_info_.setBounds(b.removeFromTop(20));
        b.removeFromTop(3);
        lbl_vst_copyright1_.setBounds(b.removeFromTop(20));
    }
  
public:
    juce::Label lbl_plugin_name_;
    juce::Label lbl_contact_info_;
    juce::Label lbl_vst_copyright1_;
    juce::Label lbl_vst_copyright2_;
};

int showModalAboutDialog(juce::Component *back_component)
{
    return juce::DialogWindow::showModalDialog("About",
                                               new AboutDialogComponent(),
                                               back_component,
                                               juce::Colours::black,
                                               true);
}
