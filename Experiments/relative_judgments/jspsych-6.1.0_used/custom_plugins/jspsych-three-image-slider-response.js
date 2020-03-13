/**
 * jspsych-three-image-slider-response
 * a jspsych plugin for free response survey questions
 *
 */


jsPsych.plugins['three-image-slider-response'] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('three-image-slider-response', 'stimulus', 'image');
  
  jsPsych.pluginAPI.registerPreload('three-image-slider-response', 'left_resp_stimulus', 'image');

  jsPsych.pluginAPI.registerPreload('three-image-slider-response', 'right_resp_stimulus', 'image');

  plugin.info = {
    name: 'three-image-slider-response',
    description: '',
    parameters: {
    trial_id: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Trial ID',
        default: undefined,
        description: 'The trial ID, for a progress bar'
      },
      
      total_trial_number: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Total number of trials',
        default: undefined,
        description: 'The total number of trials, for a progress bar'
      },
      
      stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The image to be displayed overall'
      },
      
      left_resp_stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Left response Stimulus',
        default: undefined,
        description: 'Left image to be displayed for response'
      },
      
      right_resp_stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Right response Stimulus',
        default: undefined,
        description: 'Right image to be displayed for response'
      },
      stimulus_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image height',
        default: null,
        description: 'Set the image height in pixels'
      },
      stimulus_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image width',
        default: null,
        description: 'Set the image width in pixels'
      },
      maintain_aspect_ratio: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Maintain aspect ratio',
        default: true,
        description: 'Maintain the aspect ratio after setting width or height'
      },
      
      
      resp_stimulus_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image height',
        default: null,
        description: 'Set the image height in pixels'
      },
      resp_stimulus_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image width',
        default: null,
        description: 'Set the image width in pixels'
      },
      resp_maintain_aspect_ratio: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Maintain aspect ratio',
        default: true,
        description: 'Maintain the aspect ratio after setting width or height'
      },
      
      min: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Min slider',
        default: 0,
        description: 'Sets the minimum value of the slider.'
      },
      max: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Max slider',
        default: 100,
        description: 'Sets the maximum value of the slider',
      },
      start: {
				type: jsPsych.plugins.parameterType.INT,
				pretty_name: 'Slider starting value',
				default: 50,
				description: 'Sets the starting value of the slider',
			},
      step: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Step',
        default: 1,
        description: 'Sets the step of the slider'
      },
      labels: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name:'Labels',
        default: [],
        array: true,
        description: 'Labels of the slider.',
      },
      slider_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name:'Slider width',
        default: null,
        description: 'Width of the slider in pixels.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        array: false,
        description: 'Label of the button to advance.'
      },
      check_input: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Require movement to this value',
        default: false,
        description: 'If in the slider range, the participant will have to move the slider to this value before continuing.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when user makes a response.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {

    var padding = 50;
    var html = '<div id="jspsych-three-image-slider-response-wrapper" style="margin: 0px 0px;">';
    html += trial.trial_id + ' of ' + trial.total_trial_number
    html += '<div id="jspsych-three-image-slider-response-stimulus"> ';
    html += '<img src="'+trial.stimulus+'" style="';
    if(trial.stimulus_height !== null){
      html += 'height:'+trial.stimulus_height+'px; '
      if(trial.stimulus_width == null && trial.maintain_aspect_ratio){
        html += 'width: auto; ';
      }
    }
    if(trial.stimulus_width !== null){
      html += 'width:'+trial.stimulus_width+'px; '
      if(trial.stimulus_height == null && trial.maintain_aspect_ratio){
        html += 'height: auto; ';
      }
    }
    html += '"></img>';
    html += '</div>';
    
    html += '<div id="jspsych-three-image-slider-response-stimulus">';
    
    html += '<img src="'+trial.left_resp_stimulus+'" style="';
    if(trial.resp_stimulus_height !== null){
      html += 'height:'+trial.resp_stimulus_height+'px; '
      if(trial.resp_stimulus_width == null && trial.resp_maintain_aspect_ratio){
        html += 'width: auto; ';
      }
    }
    if(trial.resp_stimulus_width !== null){
      html += 'width:'+trial.resp_stimulus_width+'px; '
      if(trial.resp_stimulus_height == null && trial.resp_maintain_aspect_ratio){
        html += 'height: auto; ';
      }
    }
    html += 'margin-right: 30px';
    html += '"></img>';
    
        html += '<img src="'+trial.right_resp_stimulus+'" style="';
    if(trial.resp_stimulus_height !== null){
      html += 'height:'+trial.resp_stimulus_height+'px; '
      if(trial.resp_stimulus_width == null && trial.resp_maintain_aspect_ratio){
        html += 'width: auto; ';
      }
    }
    if(trial.resp_stimulus_width !== null){
      html += 'width:'+trial.resp_stimulus_width+'px; '
      if(trial.resp_stimulus_height == null && trial.resp_maintain_aspect_ratio){
        html += 'height: auto; ';
      }
    }
    html += 'margin-left: 30px';
    html += '"></img>';

    html += '</div>';
    
    
    html += '<div class="jspsych-three-image-slider-response-container" style="position:relative; margin: 0 auto 3em auto; ';
    if(trial.slider_width !== null){
      html += 'width:'+trial.slider_width+'px;';
    }
    html += '">';
    html += '<input type="range" value="'+trial.start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" style="width: 100%;" id="jspsych-three-image-slider-response-response"></input>';
    html += '<div>'
    for(var j=0; j < trial.labels.length; j++){
      var width = 100/(trial.labels.length-1);
      var left_offset = (j * (100 /(trial.labels.length - 1))) - (width/2);
      html += '<div style="display: inline-block; position: absolute; left:'+left_offset+'%; text-align: center; width: '+width+'%;">';
      html += '<span style="text-align: center; font-size: 80%;">'+trial.labels[j]+'</span>';
      html += '</div>'
    }
    html += '</div>';
    html += '</div>';
    html += '</div>';

  
    html += '<output id="output"></output><br>';
    if (trial.prompt !== null){
      html += trial.prompt;
    }
    
    // add submit button, initially disabled
    html += '<button id="jspsych-three-image-slider-response-next" class="jspsych-btn">'+trial.button_label+'</button>';

    display_element.innerHTML = html;

    var response = {
      rt: null,
      response: null
    };
    
    display_element.querySelector('#jspsych-three-image-slider-response-next').disabled = true;
    
     var val = document.getElementById("jspsych-three-image-slider-response-response").value //gets the oninput value
     document.getElementById('output').innerHTML = 'You entered: ' + val //displays this value to the html page
	   
    
     // adding tracker of slider value
      display_element.querySelector('#jspsych-three-image-slider-response-response').addEventListener('change', function(){
	    var val = document.getElementById("jspsych-three-image-slider-response-response").value //gets the oninput value
	   document.getElementById('output').innerHTML = 'You entered: ' + val //displays this value to the html page
	   
	   // Enable submit
	  if(trial.check_input < 0){
	      display_element.querySelector('#jspsych-three-image-slider-response-next').disabled = false;
	  } else if (trial.check_input == val) {
	      display_element.querySelector('#jspsych-three-image-slider-response-next').disabled = false;
	  }
       
      })


    display_element.querySelector('#jspsych-three-image-slider-response-next').addEventListener('click', function() {
      // measure response time
      var endTime = performance.now();
      response.rt = endTime - startTime;
      response.response = display_element.querySelector('#jspsych-three-image-slider-response-response').value;

      if(trial.response_ends_trial){
        end_trial();
      } else {
        display_element.querySelector('#jspsych-three-image-slider-response-next').disabled = true;
      }

    });

    function end_trial(){

      jsPsych.pluginAPI.clearAllTimeouts();

      // save data
      var trialdata = {
        "rt": response.rt,
        "response": response.response
      };

      display_element.innerHTML = '';

      // next trial
      jsPsych.finishTrial(trialdata);
    }

    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#jspsych-three-image-slider-response-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

    var startTime = performance.now();
    
    
    
    
  };

  return plugin;
})();
